echo "port forwarding..."
kubectl port-forward -n allencwang service/muzero-cluster-kuberay-head-svc 8265:8265