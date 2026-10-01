from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    tokens = model.encode("Hello")

    print("TOKENS:")
    print(tokens)

    token_ids = tokens.squeeze(0).tolist()

    print("\nTOKEN IDS:")
    print(token_ids)

    logits = model.get_logits_from_input_ids(token_ids)

    print("\nNUMBER OF LOGITS:")
    print(len(logits))

    max_logit = max(logits)
    best_token_id = logits.index(max_logit)

    print("\nBEST NEXT TOKEN ID:")
    print(best_token_id)

    print("\nBEST LOGIT:")
    print(max_logit)


if __name__ == "__main__":
    main()