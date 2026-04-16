"""
HUMEM: Hebbian Universal Memory Emulation Model
Core Architecture Implementation

Author: Shreekant Jadhav
Paper: "HUMEM: Hebbian Universal Memory Emulation Model for O(1) Episodic Routing"
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class HUMEM(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", M=256, k=16, eta=0.1):
        """
        Initializes the HUMEM Dual-Process Architecture.
        
        Args:
            model_name (str): The frozen Neocortical engine.
            M (int): Number of physical neurons in the Hippocampal matrix.
            k (int): Number of winning neurons for k-WTA lateral inhibition.
            eta (float): Learning rate for unsupervised Oja's Rule updates.
        """
        super(HUMEM, self).__init__()
        
        # 1. The Neocortical Engine (Frozen LLM)
        print(f"Loading Neocortical Engine: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze the transformer weights (No backprop)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.d_model = self.encoder.config.hidden_size # Usually 2048 for Llama-1B
        
        # 2. The Artificial Hippocampus (Synaptic Matrix W_mem)
        self.M = M
        self.k = k
        self.eta = eta
        
        # Initialize W_mem randomly, then normalize to mimic biological resting state
        self.W_mem = torch.randn(self.d_model, self.M)
        self.W_mem = self.W_mem / torch.norm(self.W_mem, dim=0, keepdim=True)
        
        # 3. The Episodic Vault (Bypassing the Semantic Wall)
        self.vault = {} # Maps Winning Neuron ID (int) -> Payload (str)
        
    def extract_latent_vector(self, text):
        """
        Hijacks the frozen LLM to extract a mean-pooled latent thought vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            
        # Mean pooling across the sequence length (Tokens) to get a 1D coordinate
        embeddings = outputs.last_hidden_state # Shape: [1, seq_len, d_model]
        x = torch.mean(embeddings, dim=1).squeeze(0) # Shape: [d_model]
        return x

    def k_wta(self, activation):
        """
        k-Winner-Take-All (k-WTA) Lateral Inhibition.
        Forces sparsity by silencing all but the top k active synapses.
        """
        topk_vals, topk_indices = torch.topk(activation, self.k)
        
        y = torch.zeros_like(activation)
        y[topk_indices] = topk_vals # Only winners fire
        
        # The primary routing neuron is the absolute highest activation
        N_top = topk_indices[0].item() 
        return y, N_top

    def route(self, user_prompt):
        """
        Phase 1: O(1) Episodic Routing (Retrieval)
        Returns the retrieved payload from the vault in constant time.
        """
        x = self.extract_latent_vector(user_prompt)
        
        # O(1) Matrix Multiplication (Synaptic Distance)
        activation = torch.matmul(x, self.W_mem) 
        
        # Apply lateral inhibition
        y, N_top = self.k_wta(activation)
        
        # Retrieve payload from the Episodic Vault (bypasses Semantic Wall)
        payload = self.vault.get(N_top, None)
        return payload, x, y, N_top

    def consolidate(self, x, y, N_top, payload):
        """
        Phase 2: Unsupervised Consolidation (Learning)
        Updates the synaptic weights using Oja's Rule to prevent runaway.
        """
        # Reshape for outer product math
        x_view = x.view(-1, 1) # [d_model, 1]
        y_view = y.view(1, -1) # [1, M]
        
        # Standard Hebbian Term: neurons that fire together wire together
        hebbian_term = torch.matmul(x_view, y_view)
        
        # Oja's Normalization Decay Term
        y_squared = (y ** 2).view(1, -1)
        decay_term = self.W_mem * y_squared
        
        # Calculate Delta W
        dW = self.eta * (hebbian_term - decay_term)
        
        # Update Matrix (with a slight biological decay of 0.99 as per Algorithm 1)
        self.W_mem = (self.W_mem * 0.99) + dW
        
        # Store text in the vault
        self.vault[N_top] = payload
