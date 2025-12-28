from kubeflow.trainer import TrainerClient, CustomTrainer
import time
def simple_pytorch_training():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import os
    
    rank = int(os.getenv('RANK', 0))
    
    print("="*60)
    print(f"Node {rank}: PyTorch MNIST Training")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nNode {rank}: Using device: {device}")
    print(f"Node {rank}: PyTorch version: {torch.__version__}")
    
    # 1. Define simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    model = SimpleCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Node {rank}: Model created ({num_params:,} parameters)")
    
    # 2. Load MNIST dataset
    print(f"\nNode {rank}: Loading MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    # Use subset for quick demo
    train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True
    )
    
    print(f"Node {rank}: Dataset loaded ({len(train_dataset)} samples)")
    
    # 3. Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training loop
    print(f"\nNode {rank}: Starting training...")
    print("-"*60)
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                accuracy = 100. * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} "
                      f"Acc: {accuracy:.2f}%")
        
        epoch_acc = 100. * correct / total
        epoch_loss = total_loss / len(train_loader)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        print('='*60 + "\n")
    
    print(f"\n{'='*60}")
    print(f"Node {rank}: Training Complete!")
    print(f"Node {rank}: Final Accuracy: {epoch_acc:.2f}%")
    print('='*60)

if __name__ == "__main__":

    print("\n PyTorch MNIST Training Demo")
    print("="*80)
    time.sleep(3)
    
    # Submit
    client = TrainerClient()

    job_id = client.train(
        trainer=CustomTrainer(
            func=simple_pytorch_training,
            num_nodes=1,
            resources_per_node={
                "cpu": "500m",
                "memory": "1Gi"
            }
        )
    )
    print(f" Job ID: {job_id}\n")
    
    # Monitor
    print(" Monitoring...")
    print("="*80)
    
    for i in range(60):
        time.sleep(3)
        
        try:
            job = client.get_job(name=job_id)
            
            if not job.steps:
                print(f"[{i*3}s] Initializing...", end='\r')
                continue
            
            status = job.steps[0].status
            print(f"[{i*3}s] Status: {status}  ", end='\r')
            
            if status == "Running":
                print(f"\n\n Training started!\n")
                print(" Logs:")
                print("="*80)
                
                for log in client.get_job_logs(job_id, follow=True):
                    print(log)
                break
                
            elif status == "Succeeded":
                print(f"\n\n Completed!\n")
                for log in client.get_job_logs(job_id, follow=False):
                    print(log)
                break
                
            elif status == "Failed":
                print(f"\n\n Failed!")
                for log in client.get_job_logs(job_id, follow=False):
                    print(log)
                break
                
        except KeyboardInterrupt:
            print("\n\nInterrupted")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            break