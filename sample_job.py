from pennylane_ionq.api_client import *
from time import sleep

sample_circuit =   {
                        "qubits": 4,
                        "circuit": [
                            {
                                "gate": "h",
                                "target": 3
                            },
                            {
                                "gate": "rx",
                                "target": 0,
                                "rotation": 1
                            },
                            {
                                "gate": "cnot",
                                "target": 2,
                                "control": 0
                            },
                            {
                                "gate": "cnot",
                                "target": 1,
                                "control": 0
                            },
                            {
                                "gate": "rx",
                                "target": 0,
                                "rotation": 0.421412
                            }
                        ]
                    }

bell_circuit = \
    {
      "qubits": 2,
      "circuit": [
        {
          "gate": "h",
          "target": 0
        },
        {
          "gate": "cnot",
          "control": 0,
          "target": 1
        }
      ]
    }


sample_job = {"lang": "json", "target":"qpu",
              "body": sample_circuit}

job = Job()
job.manager.create(**sample_job)
print(job.data.value)

while not job.is_complete:
  sleep(0.01)
  print(job.status.value)
  job.reload()

job.manager.get(job.id.value)
print(job.data.value["histogram"])
