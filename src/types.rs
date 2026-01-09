use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRecordMetadata {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub id: String,
    pub values: Vec<f32>,
    pub metadata: EmbeddingRecordMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMatchMetadata {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMatch {
    pub id: String,
    pub score: f32,
    pub metadata: QueryMatchMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub matches: Vec<QueryMatch>,
}
