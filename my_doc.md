uv sync --extra all
cp -r conf.example/ conf/
# Edit conf/llm/my-model.json with your API key and model details

unzip gym_dbs.zip

podman pull shivakrishnareddyma225/enterpriseops-gym-mcp-csm:latest               
podman pull shivakrishnareddyma225/enterpriseops-gym-mcp-teams:latest               
podman pull shivakrishnareddyma225/enterpriseops-gym-mcp-email:latest               

# Restart with correct mappings (host_port:container_port)
podman run -d --name gym-csm   -p 8001:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-csm:latest
podman run -d --name gym-teams -p 8002:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-teams:latest
podman run -d --name gym-email -p 8004:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-email:latest

# To run the experiment please run evaluate.py, here is a launch.json configuration example
{
            "name": "Evaluate",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/evaluate.py",
            "args": [
                "--hf_dataset", "ServiceNow-AI/EnterpriseOps-Gym",
                "--domain", "teams",
                "--mode", "oracle",
                "--llm_config", "conf/llm/gpt-5.json",
                "--output_folder", "${input:results_folder}",
                "--orchestrator", "react",
                "--concurrency", "4",
                "--num_runs", "1"
            ],
            "env": {
                "MCP_NAME_2": "knowledge-tool-mcp",
                "MCP_ENDPOINT_2": "http://127.0.0.1:8765",
                "SYSTEM_PROMPT_SUFFIX_FILE": "${workspaceFolder}/system_prompt_knowledge_suffix.txt"
            },
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
}

# To get the final scores of the experiment, run compute_score.py, here is a launch.json configuration example:
{
            "name": "Compute Score (teams)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/compute_score.py",
            "args": [
                "--results_folder",
                "${input:results_folder}"
            ],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
}

# Stop and remove all three
podman stop gym-csm gym-teams gym-email
podman rm gym-csm gym-teams gym-email
