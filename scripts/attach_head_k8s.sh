echo "Attaching to head..."
head_container=$(kubectl get pods | grep head)
kubectl exec -it $head_container -- /bin/bash