## Use API request to retrieve prometheus metrics from OpenShift

Ensure user is logged in (do not use kubeadmin as kubeadmin does not have valid token)

```bash
$ token=`oc whoami -t` 
$ curl -G -s -k -H "Authorization: Bearer $token" 'https://thanos-querier-openshift-monitoring.apps.example.lab.redhat.com/api/v1/query?query=ALERTS' | jq -r '.data.result[] | [.metric.alertname, .metric.alertstate, .value[0], .value[1] ] | @csv'
```

```
# Output
"AlertmanagerReceiversNotConfigured","firing",1686577514.101,"1"
"ClusterNotUpgradeable","firing",1686577514.101,"1"
"KubeContainerWaiting","pending",1686577514.101,"1"
"KubePodNotReady","pending",1686577514.101,"1"
"UpdateAvailable","firing",1686577514.101,"1"
"Watchdog","firing",1686577514.101,"1"
```
