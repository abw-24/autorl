# autorl (WIP)

- Simple RL agents for use with Gym, Tensorflow 2.x
- Currently only basic value agents (no explicit policy representation) implemented
- Test suite should be run with `pytest` as the test runner:
```
pytest src/autorl --disable-warnings
```
    - Note: Integration tests do not mock data or writing to disk, so the full suite may take a few minutes to run