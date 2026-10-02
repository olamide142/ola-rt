use std::collections::BTreeMap;

use ola_core::{ArchitectureId, ModelDescriptor};

use crate::{ArchitectureLoadError, ModelDefinition, ModelError, ModelSource};

/// Constructs one implementation-specific model type from a definition.
///
/// The generic output `M` keeps architecture selection independent of Burn or
/// any future execution backend. A registry contains factories producing one
/// common output type chosen by its owner.
pub trait ArchitectureFactory<M>: Send + Sync {
    /// Identifies the architecture implemented by this factory.
    fn architecture(&self) -> &ArchitectureId;

    /// Resolves configuration and weights and constructs a model instance.
    fn load(&self, definition: &ModelDefinition) -> Result<M, ArchitectureLoadError>;
}

/// Architecture implementations available for one output model type `M`.
pub struct ArchitectureRegistry<M> {
    factories: BTreeMap<ArchitectureId, Box<dyn ArchitectureFactory<M>>>,
}

impl<M> Default for ArchitectureRegistry<M> {
    fn default() -> Self {
        Self {
            factories: BTreeMap::new(),
        }
    }
}

impl<M> ArchitectureRegistry<M> {
    /// Registers a factory under its architecture ID.
    pub fn register<F>(&mut self, factory: F) -> Result<(), ModelError>
    where
        F: ArchitectureFactory<M> + 'static,
    {
        let architecture = factory.architecture().clone();
        if architecture.is_empty() {
            return Err(ModelError::InvalidArchitecture(architecture));
        }
        if self.factories.contains_key(&architecture) {
            return Err(ModelError::DuplicateArchitecture(architecture));
        }
        self.factories.insert(architecture, Box::new(factory));
        Ok(())
    }

    /// Constructs the model by selecting the factory named in its descriptor.
    pub fn load(&self, definition: &ModelDefinition) -> Result<LoadedModel<M>, ModelError> {
        definition.validate()?;
        let descriptor = &definition.descriptor;
        let architecture = &descriptor.architecture;
        let factory = self.factories.get(architecture).ok_or_else(|| {
            ModelError::UnsupportedArchitecture {
                model: descriptor.id.clone(),
                architecture: architecture.clone(),
            }
        })?;
        let inner = factory
            .load(definition)
            .map_err(|source| ModelError::Load {
                model: descriptor.id.clone(),
                architecture: architecture.clone(),
                source,
            })?;

        Ok(LoadedModel {
            descriptor: descriptor.clone(),
            source: definition.source.clone(),
            inner,
        })
    }

    /// Returns whether this registry can construct an architecture.
    pub fn supports(&self, architecture: &ArchitectureId) -> bool {
        self.factories.contains_key(architecture)
    }

    /// Iterates supported architecture IDs in stable order.
    pub fn architectures(&self) -> impl ExactSizeIterator<Item = &ArchitectureId> {
        self.factories.keys()
    }

    /// Returns the number of registered architecture implementations.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Returns whether no architecture implementation is registered.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }
}

/// Constructed model paired with the metadata used to create it.
///
/// Retaining provenance alongside the implementation is important for traces,
/// benchmark reports, and runtime registration.
#[derive(Debug)]
pub struct LoadedModel<M> {
    descriptor: ModelDescriptor,
    source: ModelSource,
    inner: M,
}

impl<M> LoadedModel<M> {
    /// Returns backend-neutral model metadata.
    pub fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    /// Returns the artifact source used during construction.
    pub fn source(&self) -> &ModelSource {
        &self.source
    }

    /// Returns the implementation-specific model instance.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Mutably returns the implementation-specific model instance.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// Consumes the wrapper and returns the implementation-specific instance.
    pub fn into_inner(self) -> M {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use ola_core::{ArchitectureId, ModelDescriptor, ModelId};

    use super::*;
    use crate::ModelCatalog;

    struct StubFactory {
        architecture: ArchitectureId,
        implementation: &'static str,
    }

    impl ArchitectureFactory<String> for StubFactory {
        fn architecture(&self) -> &ArchitectureId {
            &self.architecture
        }

        fn load(&self, definition: &ModelDefinition) -> Result<String, ArchitectureLoadError> {
            Ok(format!(
                "{}:{}",
                self.implementation, definition.descriptor.id
            ))
        }
    }

    fn definition(model: &str, architecture: &str) -> ModelDefinition {
        ModelDefinition::local(
            ModelDescriptor::language(model, architecture),
            format!("models/{model}"),
        )
    }

    #[test]
    fn different_models_share_one_architecture_factory() {
        let mut catalog = ModelCatalog::default();
        catalog.register(definition("tinyllama", "llama")).unwrap();
        catalog.register(definition("llama-1b", "llama")).unwrap();
        let mut architectures = ArchitectureRegistry::default();
        architectures
            .register(StubFactory {
                architecture: "llama".into(),
                implementation: "burn-llama",
            })
            .unwrap();

        let first = architectures
            .load(catalog.get(&ModelId::from("tinyllama")).unwrap())
            .unwrap();
        let second = architectures
            .load(catalog.get(&ModelId::from("llama-1b")).unwrap())
            .unwrap();

        assert_eq!(first.inner(), "burn-llama:tinyllama");
        assert_eq!(second.inner(), "burn-llama:llama-1b");
    }

    #[test]
    fn selects_different_factories_by_architecture() {
        let mut architectures = ArchitectureRegistry::default();
        for (architecture, implementation) in [("llama", "burn-llama"), ("qwen2", "burn-qwen")] {
            architectures
                .register(StubFactory {
                    architecture: architecture.into(),
                    implementation,
                })
                .unwrap();
        }

        let loaded = architectures
            .load(&definition("qwen-small", "qwen2"))
            .unwrap();

        assert_eq!(loaded.inner(), "burn-qwen:qwen-small");
    }

    #[test]
    fn reports_an_unsupported_architecture_without_fallback() {
        let architectures = ArchitectureRegistry::<String>::default();

        let error = architectures
            .load(&definition("mystery", "unknown"))
            .unwrap_err();

        assert_eq!(
            error,
            ModelError::UnsupportedArchitecture {
                model: "mystery".into(),
                architecture: "unknown".into(),
            }
        );
    }
}
