use async_openai::config::OpenAIConfig;
use async_openai::types::CreateEmbeddingRequestArgs;
use async_openai::Client as OpenAIClient;
use thiserror::Error;

use crate::constants::EMBEDDING_MODEL;
use crate::env_load::Config;
use crate::pinecone_client::{PineconeClient, PineconeError};
use crate::types::{QueryMatch, QueryMatchMetadata, QueryResult};

#[derive(Error, Debug)]
pub enum QueryError {
    #[error("OpenAI error: {0}")]
    OpenAI(#[from] async_openai::error::OpenAIError),
    #[error("Pinecone error: {0}")]
    Pinecone(#[from] PineconeError),
}

pub async fn query_db(prompt: &str, config: &Config) -> Result<QueryResult, QueryError> {
    let openai_config = OpenAIConfig::new().with_api_key(&config.openai_api_key);
    let openai_client = OpenAIClient::with_config(openai_config);

    let embedding_request = CreateEmbeddingRequestArgs::default()
        .model(EMBEDDING_MODEL)
        .input([prompt])
        .build()?;

    let embedding_response = openai_client.embeddings().create(embedding_request).await?;
    let embedding: Vec<f32> = embedding_response.data[0].embedding.clone();

    let pinecone = PineconeClient::new(&config.pinecone_api_key);
    let query_response = pinecone
        .query(&config.pinecone_index_name, &embedding, 5, true)
        .await?;

    let matches: Vec<QueryMatch> = query_response
        .matches
        .iter()
        .map(|m| QueryMatch {
            id: m.id.clone(),
            score: m.score,
            metadata: QueryMatchMetadata {
                text: m
                    .metadata
                    .as_ref()
                    .and_then(|md| md.get("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            },
        })
        .collect();

    Ok(QueryResult { matches })
}
