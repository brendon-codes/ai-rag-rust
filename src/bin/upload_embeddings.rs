use std::fs;

use rag::env_load::Config;
use rag::pinecone_client::PineconeClient;
use rag::types::EmbeddingRecord;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;

    let json = fs::read_to_string("data/embeddings/embedding.json")?;
    let embeddings: Vec<EmbeddingRecord> = serde_json::from_str(&json)?;

    let pinecone = PineconeClient::new(&config.pinecone_api_key);
    pinecone
        .upsert(&config.pinecone_index_name, &embeddings)
        .await?;

    println!("Uploaded {} vectors", embeddings.len());

    Ok(())
}
