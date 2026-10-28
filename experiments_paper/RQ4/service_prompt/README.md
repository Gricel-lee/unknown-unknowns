# RQ4: online services for the most frequent concepts

Once an LLM has hypothesised a candidate parameter, the system still needs a way to observe it.
This experiment asks Gemini whether online (web) services can provide the **two most frequent
concepts** of each case study (from `../results/concepts.csv`, column
`concept_authors_classified`) to a system that cannot measure them itself.

| Case study | Concept 1 | Concept 2 |
|---|---|---|
| Robot | terrain condition | crop condition |
| Consensus | server load | network congestion |
| FX | system load | market volatility |
| UAV | camera calibration | image/scene complexity |

## Contents

```
service_prompt/
├── case_studies_prompts/   <case>_prompt.txt: one prompt per case study, asking for both concepts
├── responses/              <case>.md: Gemini's answer
├── services.csv            one row per query: case, model that answered, date, prompt, response
└── query_services.py       sends the prompts and saves the answers
```

Each prompt follows the template *"I would like my &lt;system&gt; to invoke online (web) services
helping it to estimate (1) &lt;concept 1&gt; and (2) &lt;concept 2&gt; ..., as the &lt;system&gt;
doesn't have sensors for these onboard. Can you recommend any such services?"*

## Running

The queries use the Gemini model and settings of `../query.py` (`gemini-3.5-flash` through
Vertex AI, queried on 2026-10-25). With the Gemini environment variables of `../query.py` set,
run from the repository root:

```bash
python experiments_paper/RQ4/service_prompt/query_services.py
```

Answers that already exist in `responses/` are skipped; delete one to query it again.

## Summary of the answers

Gemini recommended services for all eight concepts. Whether each recommendation provides the
concept is the authors' reading of the answers; the services themselves have not been verified.

| Case study | Concept | Result | Recommended services |
|---|---|---|---|
| Robot | terrain condition | Found | Open-Meteo and AgroMonitoring (soil moisture at GPS coordinates), ISRIC / SoilWeb (soil type) |
| Robot | crop condition | Found | AgroMonitoring, Sentinel Hub, Planet (vegetation indices such as NDVI) |
| Consensus | server load | Self-measured | Datadog, AWS CloudWatch, Grafana Cloud (require monitoring agents on the servers) |
| Consensus | network congestion | Self-measured | ThousandEyes (agents on the controller and servers); Pingdom / UptimeRobot measure from their own locations only |
| FX | system load | Self-measured | Cloud monitoring APIs, Datadog, Netdata Cloud (require monitoring agents) |
| FX | market volatility | Found | Twelve Data, Alpha Vantage (ready-made volatility indicators, e.g., ATR), OANDA |
| UAV | camera calibration | Proxy | Sightengine, Everypixel (image sharpness and exposure; true calibration is not supported) |
| UAV | image/scene complexity | Proxy | Roboflow, AWS Rekognition, Google Cloud Vision (detected objects, from which complexity is derived) |

- **Found** (3): an external service provides the concept directly.
- **Proxy** (2): an external service provides related data from which the concept can be estimated.
- **Self-measured** (3): the concept describes the system's own state, so it can only be obtained
  by instrumenting the system itself.
- **Not found** (0).

Environmental concepts (soil, crops, markets) can be obtained from external services, whereas
the internal state of a system (load, congestion) can only be measured by the system itself.
