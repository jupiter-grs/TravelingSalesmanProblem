# TravelingSalesmanProblem

TSP with [Amplify](https://amplify.fixstars.com/) and [D-Wave](https://www.dwavesys.com/).

## Set up
Install packages.
```sh
pip install -r requirements.txt
```

Create .env file.
```.env
AMPLIFY=<YOUR TOKEN>
```

## Run
Amplify
```sh
python amplify_tsp.py
```

D-Wave (modeled with PyQUBO)
```sh
python pyqubo_tsp.py
```
