# AWS EC2 Deployment Guide

This guide covers how to launch, configure and run the DS-Agent benchmark suite on an AWS EC2 GPU instance.

---

## Instance Configuration

| Setting | Value |
|---|---|
| **AMI** | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5.1 (Ubuntu 22.04) |
| **Instance Type** | `g4dn.xlarge`

---

## Setup Instructions

### 1. SSH into the Instance

```bash
ssh -i "/path/to/your-key.pem" ubuntu@<EC2_PUBLIC_IP>
```

### 2. Activate the PyTorch Environment

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch
```

### 3. Clone the Repository

```bash
git clone https://github.com/muhammadhaider02/DS-Agent.git
```

### 4. Install Dependencies

```bash
cd DS-Agent
pip install -r requirements.txt

cd development
pip install -e .
```

### 5. Upload Required Data Folders

Run the following from a **separate local terminal** (Git Bash / PowerShell):

```bash
# benchmarks/ is gitignored (large files). Download from Google Drive as instructed in README.md,
# unzip as DS-Agent/development/MLAgentBench/benchmarks, then upload to EC2.
scp -i "/path/to/your-key.pem" -r "/local/path/to/DS-Agent/development/MLAgentBench/benchmarks" ubuntu@<EC2_PUBLIC_IP>:~/DS-Agent/development/MLAgentBench/

# data/ is distributed as data.zip (see README.md). Extract it locally as DS-Agent/development/data, then upload to EC2.
scp -i "/path/to/your-key.pem" -r "/local/path/to/DS-Agent/development/data" ubuntu@<EC2_PUBLIC_IP>:~/DS-Agent/development/
```

### 6. Create the `.env` File

```bash
cat > .env << 'EOF'
DEEPSEEK_API_KEY=<your_deepseek_api_key>
DEEPSEEK_BASE_URL=https://api.deepseek.com
HF_TOKEN=<your_huggingface_token>
EOF
```

### 7. Verify Setup

```bash
cat .env
ls ~/DS-Agent/development/data
ls ~/DS-Agent/development/MLAgentBench/benchmarks
```

### 8. Authenticate and Pre-download the Embedding Model

```bash
# Use the same HF_TOKEN value from your .env file
hf auth login --token <your_huggingface_token>
hf download BAAI/llm-embedder
```

---

## Running the Benchmark

```bash
cd ~/DS-Agent/development/MLAgentBench
chmod +x run_all.sh

nohup ./run_all.sh > benchmark_output.log 2>&1 &
```

---

## Monitoring

```bash
# Check which tasks have started and when
grep "Task:" benchmark_output.log

# Watch live output
tail -f benchmark_output.log

# View agent reasoning in real time
tail -f logs/$(ls -t logs/ | head -1)/agent_log/main_log

# Check running processes
ps aux | grep runner.py
```

---

## Downloading Results

Run from your local machine after all runs complete:

```bash
scp -r -i "/path/to/your-key.pem" ubuntu@<EC2_PUBLIC_IP>:~/DS-Agent/development/MLAgentBench/logs/ ./results/logs/
scp -r -i "/path/to/your-key.pem" ubuntu@<EC2_PUBLIC_IP>:~/DS-Agent/development/MLAgentBench/workspace/ ./results/workspace/
scp -i "/path/to/your-key.pem" ubuntu@<EC2_PUBLIC_IP>:~/DS-Agent/development/MLAgentBench/benchmark_output.log ./results/
```
