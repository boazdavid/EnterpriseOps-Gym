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

# Stop and remove all three
podman stop gym-csm gym-teams gym-email
podman rm gym-csm gym-teams gym-email

python ray_experiment_queue.py --experiment_config conf/ray/experiment.json

python compute_score.py --results_folder results/react/my-model/teams