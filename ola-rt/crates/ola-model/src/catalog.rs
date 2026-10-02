use std::collections::BTreeMap;

use ola_core::ModelId;

use crate::{ModelDefinition, ModelError};

/// Deterministically ordered collection of selectable model definitions.
#[derive(Clone, Debug, Default)]
pub struct ModelCatalog {
    definitions: BTreeMap<ModelId, ModelDefinition>,
}

impl ModelCatalog {
    /// Registers a validated definition, rejecting duplicate model IDs.
    pub fn register(&mut self, definition: ModelDefinition) -> Result<(), ModelError> {
        definition.validate()?;
        let id = definition.descriptor.id.clone();
        if self.definitions.contains_key(&id) {
            return Err(ModelError::DuplicateModel(id));
        }
        self.definitions.insert(id, definition);
        Ok(())
    }

    /// Selects one definition by model ID.
    pub fn get(&self, id: &ModelId) -> Result<&ModelDefinition, ModelError> {
        self.definitions
            .get(id)
            .ok_or_else(|| ModelError::UnknownModel(id.clone()))
    }

    /// Iterates definitions in stable model-ID order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ModelDefinition> {
        self.definitions.values()
    }

    /// Returns the number of selectable definitions.
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Returns whether the catalog contains no definitions.
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }
}
