from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch
import os

class SimpleFineTuner:
    """LoRA on FLAN-T5"""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.finetuned_model_path = "./finetuned_models/flan-t5-lora"
        
    def prepare_model(self):
        """ LoRA adapters"""
        print("Loading base model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )
        
        print("Adding LoRA adapters...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def create_dataset(self, examples: list):
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples["input"],
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            
            labels = self.tokenizer(
                examples["output"],
                max_length=128,
                truncation=True,
                padding="max_length"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        dataset = Dataset.from_dict({
            "input": [ex["input"] for ex in examples],
            "output": [ex["output"] for ex in examples]
        })
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, train_examples: list, epochs: int = 3):
        """Fine-tune the model"""
        
        if self.model is None:
            self.prepare_model()
        
        print(f"Preparing dataset with {len(train_examples)} examples...")
        train_dataset = self.create_dataset(train_examples)
        
        print("Setting up training...")
        training_args = TrainingArguments(
            output_dir="./finetuning_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            learning_rate=3e-4,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Saving fine-tuned model...")
        os.makedirs(self.finetuned_model_path, exist_ok=True)
        self.model.save_pretrained(self.finetuned_model_path)
        self.tokenizer.save_pretrained(self.finetuned_model_path)
        
        print(f" Model saved to {self.finetuned_model_path}")
        
        return self.model
    
    def load_finetuned_model(self):
        """Load the fine-tuned model"""
        if not os.path.exists(self.finetuned_model_path):
            print("No fine-tuned model found. Please train first.")
            return None
        
        print("Loading fine-tuned model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base_model, self.finetuned_model_path)
        
        print("✅ Fine-tuned model loaded!")
        return self.model
    
    def generate(self, prompt: str, max_length: int = 128):
        """Generate text with the model"""
        if self.model is None:
            self.load_finetuned_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

finetuner = SimpleFineTuner()