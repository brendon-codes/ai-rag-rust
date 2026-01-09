use std::env;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EnvError {
    #[error("Environment variable {0} is not set")]
    Missing(String),
    #[error("Failed to parse {0}: {1}")]
    Parse(String, String),
}

pub struct Config {
    pub openai_api_key: String,
    pub pinecone_api_key: String,
    pub pinecone_index_name: String,
    pub pinecone_index_dimension: u32,
    pub pinecone_index_metric: String,
    pub pinecone_index_cloud: String,
    pub pinecone_index_region: String,
}

impl Config {
    pub fn from_env() -> Result<Self, EnvError> {
        dotenvy::dotenv().ok();

        let openai_api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| EnvError::Missing("OPENAI_API_KEY".to_string()))?;

        let pinecone_api_key = env::var("PINECONE_API_KEY")
            .map_err(|_| EnvError::Missing("PINECONE_API_KEY".to_string()))?;

        let pinecone_index_name = env::var("PINECONE_INDEX_NAME")
            .map_err(|_| EnvError::Missing("PINECONE_INDEX_NAME".to_string()))?;

        let pinecone_index_dimension: u32 = env::var("PINECONE_INDEX_DIMENSION")
            .map_err(|_| EnvError::Missing("PINECONE_INDEX_DIMENSION".to_string()))?
            .parse()
            .map_err(|e| {
                EnvError::Parse("PINECONE_INDEX_DIMENSION".to_string(), format!("{}", e))
            })?;

        let pinecone_index_metric =
            env::var("PINECONE_INDEX_METRIC").unwrap_or_else(|_| "cosine".to_string());

        let pinecone_index_cloud = env::var("PINECONE_INDEX_CLOUD")
            .map_err(|_| EnvError::Missing("PINECONE_INDEX_CLOUD".to_string()))?;

        let pinecone_index_region = env::var("PINECONE_INDEX_REGION")
            .map_err(|_| EnvError::Missing("PINECONE_INDEX_REGION".to_string()))?;

        Ok(Self {
            openai_api_key,
            pinecone_api_key,
            pinecone_index_name,
            pinecone_index_dimension,
            pinecone_index_metric,
            pinecone_index_cloud,
            pinecone_index_region,
        })
    }
}
