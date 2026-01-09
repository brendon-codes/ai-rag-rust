use rag::env_load::Config;
use rag::pinecone_client::PineconeClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let pinecone = PineconeClient::new(&config.pinecone_api_key);

    let exists = pinecone.index_exists(&config.pinecone_index_name).await?;

    if exists {
        return Err(format!("{} already exists", config.pinecone_index_name).into());
    }

    pinecone
        .create_index(
            &config.pinecone_index_name,
            config.pinecone_index_dimension,
            &config.pinecone_index_metric,
            &config.pinecone_index_cloud,
            &config.pinecone_index_region,
        )
        .await?;

    println!("Created index: {}", config.pinecone_index_name);

    Ok(())
}
