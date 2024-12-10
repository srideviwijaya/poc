
**Create Service**
```
apiVersion: v1
kind: Service
metadata:
  labels:
    amd-node-service: "true"
  name: amd-node-service
  namespace: llm-test
spec:
  ports:
    - name: metrics
      port: 9100
      protocol: TCP
      targetPort: 9100
      nodePort: 0
  selector: {}
  type: ClusterIP
```


**Create Endpoints**
```
kind: Endpoints
apiVersion: v1
metadata:
  name: amd-node-service
  namespace: llm-test
subsets:
  - addresses:
    - ip: "xxxx"
    ports:
      - port: 9100
        name: metrics
```

**Create ServiceMonitor**
```
apiVersion: 1
kind: ServiceMonitor
metadata:
  name: amd-service-monitor
  namespace: llm-test
spec:
  endpoints:
  - interval: 15s
    port: metrics
    selector:
      matchLabels:
        amd-node-service: "true"
```

