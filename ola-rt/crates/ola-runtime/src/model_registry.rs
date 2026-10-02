use std::collections::BTreeMap;

use ola_core::{ModelDescriptor, ModelId};

use crate::RuntimeError;

#[derive(Clone, Debug, Default)]
/// Collection of models visible to one runtime.
pub struct ModelRegistry {
    models: BTreeMap<ModelId, ModelDescriptor>,
}

impl ModelRegistry {
    /// Registers a model, rejecting duplicate identifiers.
    pub fn register(&mut self, model: ModelDescriptor) -> Result<(), RuntimeError> {
        model.validate()?;
        if self.models.contains_key(&model.id) {
            return Err(RuntimeError::DuplicateModel(model.id));
        }
        self.models.insert(model.id.clone(), model);
        Ok(())
    }

    /// Looks up model metadata by stable identifier.
    pub fn get(&self, id: &ModelId) -> Option<&ModelDescriptor> {
        self.models.get(id)
    }

    /// Returns whether an identifier is registered.
    pub fn contains(&self, id: &ModelId) -> bool {
        self.models.contains_key(id)
    }

    /// Iterates in stable model-ID order for deterministic output.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ModelDescriptor> {
        self.models.values()
    }

    /// Returns the number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns whether no models are registered.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}
