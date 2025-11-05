from app.finetuning.trainer import finetuner
from app.finetuning.training_data import get_training_data

def main():
    print(" Starting Fine-Tuning Process")
    print("=" * 60)
    
    training_examples = get_training_data()
    print(f" Training examples: {len(training_examples)}")
    
    print("\n Fine-tuning FLAN-T5 with LoRA")
    finetuner.train(training_examples, epochs=3)
    
    print("\n Fine-tuning complete!")
    print(f" Model saved to: {finetuner.finetuned_model_path}")
    
    print("\n Testing fine-tuned model...")
    test_prompt = "Question: Who is Klajdi's best friend? Context: Helios and Klajdi are best friends."
    result = finetuner.generate(test_prompt)
    print(f"Test result: {result}")

if __name__ == "__main__":
    main()