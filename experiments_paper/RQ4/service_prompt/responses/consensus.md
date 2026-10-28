To achieve this, the controller needs to query external Web APIs that aggregate telemetry from your servers. Because of security and network topology, **no generic public web service can measure your private servers' load or the specific network path between your controller and servers without some setup.**

Instead, you must use **Observability-as-a-Service (SaaS)** platforms or **Cloud Provider APIs**. Your servers and network infrastructure will stream metrics to these services, and your controller will query their web APIs to make consensus decisions.

Here are the best service recommendations categorized by your two requirements.

---

### Part 1: Services to Estimate Server Load (CPU, RAM, Queue Depth)

To get server load, the servers must report their metrics to a cloud telemetry service, which your controller then queries via a REST API.

#### 1. Cloud Provider APIs (AWS CloudWatch / GCP Cloud Monitoring)
If your servers are hosted on AWS, Google Cloud, or Azure, the cloud providers themselves are the best web services. They collect hypervisor-level metrics automatically without installing software.
*   **How it works:** The controller makes an HTTPS request to the CloudWatch/GCP API.
*   **Key Metric to Query:** `CPUUtilization`, `MemoryUtilization`, or `NetworkIn`/`NetworkOut`.
*   **Pros:** Highly reliable, already integrated into your infrastructure, virtually zero-latency API responses.
*   **Cons:** Metrics can sometimes be delayed by 1–5 minutes unless you pay for "detailed monitoring" (1-second or 10-second resolution).

#### 2. Datadog (Query API)
Datadog is a leading SaaS monitoring platform. You install a lightweight agent on your two servers, which stream metrics to Datadog's cloud.
*   **How it works:** Your controller calls the Datadog HTTP API: `GET /api/v1/query` with a query like `system.cpu.idle{host:server_a}`.
*   **Pros:** Near real-time metrics (sub-10-second latency), highly detailed host metrics, excellent API documentation.
*   **Cons:** Paid service, requires agent installation on the servers.

#### 3. Prometheus + Grafana Cloud (Hosted API)
If you prefer open-source standards, you can run Prometheus exporters on your servers and have them scrape metrics to Grafana Cloud.
*   **How it works:** The controller queries the Grafana Cloud Prometheus-compatible API using PromQL (Prometheus Query Language).
*   **Pros:** Industry standard for distributed systems, highly customizable metrics.

---

### Part 2: Services to Estimate Network Congestion

Measuring network congestion between the **controller** and the **servers** via a third party is tricky. If a third-party service pings Server A, it measures the latency between *the service* and Server A, not between *your controller* and Server A. 

To measure the actual path between the Controller and the Servers, you need these services:

#### 1. Cisco ThousandEyes (Network Intelligence Service)
ThousandEyes is the industry leader for WAN and Internet path monitoring. 
*   **How it works:** You install a "Constraint Agent" (a lightweight container) on your controller and your servers. ThousandEyes continuously maps the network path between them, measuring packet loss, jitter, latency, and BGP routing changes.
*   **How the controller uses it:** The controller queries the ThousandEyes v2 API to get the current latency/loss between the controller's agent and the servers' agents.
*   **Pros:** Detects actual internet routing congestion, ISP outages, and packet loss.
*   **Cons:** Expensive enterprise tool.

#### 2. Pingdom / Uptime Robot APIs (Simplest Proxy)
If you cannot install agents and just want a general "is the network to Server A congested?" check.
*   **How it works:** These services ping your servers from various global locations. 
*   **How the controller uses it:** The controller queries their API to get the latest response time. 
*   **Pros:** Easy to set up; no agents required on the controller.
*   **Cons:** It measures *global* network congestion to your servers, not the specific route from your controller.

---

### Recommended Architecture: The "Cloud-Native" Approach

If your infrastructure is hosted in a public cloud (e.g., AWS), here is the most robust, low-latency setup:

```
                  ┌──────────────────────┐
                  │  CloudWatch / GCP    │
                  │  Monitoring Cloud    │
                  └──────────▲───────────┘
                             │ (Query API)
                             │
  ┌──────────────┐    ┌──────┴───────┐    ┌──────────────┐
  │   Server A   │◄───┤  Controller  ├───►│   Server B   │
  │ (Consensus)  │    └──────────────┘    │ (Consensus)  │
  └──────────────┘                        └──────────────┘
```

1.  **For Server Load:** Use **AWS CloudWatch API** or **Datadog API**. The controller queries this every few seconds to get the CPU load of Server A vs. Server B.
2.  **For Network Congestion:** Have the Controller run a background thread that performs a simple TCP Handshake or ICMP Ping to Server A and Server B once every second, saving the result to local memory. *(Even though you stated the controller "doesn't measure these itself," relying on a third-party web service for real-time network latency introduces more latency than the network congestion it is trying to measure).* 

If a local ping is strictly impossible, **ThousandEyes** is the gold standard web service for mapping network path congestion between specific nodes.