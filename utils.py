
def config_inference():
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Snowflake/Arctic-Text2SQL-R1-7B")
    print(config.max_position_embeddings)

if __name__ == "__main__":
    config_inference()
