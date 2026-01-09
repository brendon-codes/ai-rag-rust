use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::EmbeddingRecord;

#[derive(Error, Debug)]
pub enum PineconeError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error: {0}")]
    Api(String),
    #[error("Index not found: {0}")]
    IndexNotFound(String),
}

#[derive(Serialize)]
struct CreateIndexRequest {
    name: String,
    dimension: u32,
    metric: String,
    spec: ServerlessSpec,
}

#[derive(Serialize)]
struct ServerlessSpec {
    serverless: ServerlessConfig,
}

#[derive(Serialize)]
struct ServerlessConfig {
    cloud: String,
    region: String,
}

#[derive(Serialize)]
struct UpsertRequest {
    vectors: Vec<VectorRecord>,
}

#[derive(Serialize)]
struct VectorRecord {
    id: String,
    values: Vec<f32>,
    metadata: serde_json::Value,
}

#[derive(Serialize)]
struct QueryRequest {
    vector: Vec<f32>,
    #[serde(rename = "topK")]
    top_k: u32,
    #[serde(rename = "includeMetadata")]
    include_metadata: bool,
}

#[derive(Deserialize)]
pub struct QueryResponse {
    pub matches: Vec<PineconeMatch>,
}

#[derive(Deserialize)]
pub struct PineconeMatch {
    pub id: String,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct DescribeIndexResponse {
    host: String,
}

pub struct PineconeClient {
    client: Client,
    api_key: String,
}

impl PineconeClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.to_string(),
        }
    }

    pub async fn index_exists(&self, name: &str) -> Result<bool, PineconeError> {
        let url = format!("https://api.pinecone.io/indexes/{}", name);

        let response = self
            .client
            .get(&url)
            .header("Api-Key", &self.api_key)
            .header("Accept", "application/json")
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    pub async fn create_index(
        &self,
        name: &str,
        dimension: u32,
        metric: &str,
        cloud: &str,
        region: &str,
    ) -> Result<(), PineconeError> {
        let url = "https://api.pinecone.io/indexes";

        let request = CreateIndexRequest {
            name: name.to_string(),
            dimension,
            metric: metric.to_string(),
            spec: ServerlessSpec {
                serverless: ServerlessConfig {
                    cloud: cloud.to_string(),
                    region: region.to_string(),
                },
            },
        };

        let response = self
            .client
            .post(url)
            .header("Api-Key", &self.api_key)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PineconeError::Api(format!(
                "Failed to create index: {} - {}",
                status, body
            )));
        }

        Ok(())
    }

    async fn get_index_host(&self, name: &str) -> Result<String, PineconeError> {
        let url = format!("https://api.pinecone.io/indexes/{}", name);

        let response = self
            .client
            .get(&url)
            .header("Api-Key", &self.api_key)
            .header("Accept", "application/json")
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(PineconeError::IndexNotFound(name.to_string()));
        }

        let index_info: DescribeIndexResponse = response.json().await?;
        Ok(index_info.host)
    }

    pub async fn upsert(
        &self,
        index_name: &str,
        records: &[EmbeddingRecord],
    ) -> Result<(), PineconeError> {
        let host = self.get_index_host(index_name).await?;
        let url = format!("https://{}/vectors/upsert", host);

        let vectors: Vec<VectorRecord> = records
            .iter()
            .map(|r| VectorRecord {
                id: r.id.clone(),
                values: r.values.clone(),
                metadata: serde_json::json!({ "text": r.metadata.text }),
            })
            .collect();

        let request = UpsertRequest { vectors };

        let response = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PineconeError::Api(format!(
                "Failed to upsert: {} - {}",
                status, body
            )));
        }

        Ok(())
    }

    pub async fn query(
        &self,
        index_name: &str,
        vector: &[f32],
        top_k: u32,
        include_metadata: bool,
    ) -> Result<QueryResponse, PineconeError> {
        let host = self.get_index_host(index_name).await?;
        let url = format!("https://{}/query", host);

        let request = QueryRequest {
            vector: vector.to_vec(),
            top_k,
            include_metadata,
        };

        let response = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(PineconeError::Api(format!(
                "Failed to query: {} - {}",
                status, body
            )));
        }

        let query_response: QueryResponse = response.json().await?;
        Ok(query_response)
    }
}
