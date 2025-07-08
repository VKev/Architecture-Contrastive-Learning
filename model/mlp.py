import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for Iris dataset classification.
    
    Architecture:
    - Input: 4 features (sepal length, sepal width, petal length, petal width)
    - Hidden Layer 1: 16 neurons with ReLU activation
    - Hidden Layer 2: 8 neurons with ReLU activation
    - Output: 3 classes (setosa, versicolor, virginica)
    """
    
    def __init__(self, input_size=4, hidden1_size=64, hidden2_size=32, num_classes=3, dropout_rate=0.0):
        super(SimpleMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.num_classes = num_classes
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 4)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 3)
        """
        # Flatten input if needed (in case input has extra dimensions)
        x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        """
        Make predictions with the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x):
        """
        Get prediction probabilities.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


def create_simple_mlp(input_size=4, hidden1_size=16, hidden2_size=8, num_classes=3, dropout_rate=0.2):
    """
    Factory function to create a SimpleMLP instance.
    
    Args:
        input_size (int): Number of input features (default: 4 for Iris)
        hidden1_size (int): Size of first hidden layer (default: 16)
        hidden2_size (int): Size of second hidden layer (default: 8)
        num_classes (int): Number of output classes (default: 3 for Iris)
        dropout_rate (float): Dropout rate for regularization (default: 0.2)
        
    Returns:
        SimpleMLP: Initialized MLP model
    """
    return SimpleMLP(input_size, hidden1_size, hidden2_size, num_classes, dropout_rate)


if __name__ == "__main__":
    # Example usage and testing
    model = SimpleMLP()
    print("SimpleMLP Architecture:")
    print(model)
    
    # Test with dummy data
    batch_size = 32
    dummy_input = torch.randn(batch_size, 4)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Predictions
    predictions = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    
    # Probabilities
    probabilities = model.predict_proba(dummy_input)
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
