# Yassir Eta Prediction Shiny App

Yassir— a ride-hailing app, depend heavily on real-time data and machine learning algorithms to automate and optimize their services. Accurate prediction of the Estimated Time of Arrival (ETA) is crucial for enhancing the reliability and attractiveness of Yassir's services.  This shiny application embeds ML models for accurate prediction of ETA.

<a name="readme-top"></a>

## Framework

The CRoss Industry Standard Process for Data Mining (CRISP-DM).

## Data dictionary

1. `Timestamp:` Time that the trip was started
2. `Origin_lat:` Origin latitude (in degrees)
3. `Origin_lon:` Origin longitude (in degrees)
4. `Destination_lat:` Destination latitude (in degrees)
5. `Destination_lon:` Destination longitude (in degrees)
6. `Trip_distance:` Distance in meters on a driving route
7. **ETA prediction:** Estimated trip time in seconds

### Notebook size too big for GitHub?

- Explore visualization on Google Colab [ETA.ipynb](https://colab.research.google.com/drive/1cqx0GfZikrG0wSBtvn7sEzrfwmo8H67b#scrollTo=QaXeP7oiXnaQ)

### FastAPI backend

[API](https://gabcares-yassir-eta-api.hf.space/docs)

[FastAPI Image](https://hub.docker.com/repository/docker/gabcares/yassir-eta-fastapi)

#### Demo video- FastAPI (RESTFul)

- Coming soon!

### Shiny for Python frontend application

[client](https://gabcares.shinyapps.io/yassir-eta-data-app/)

[Shiny Image](https://hub.docker.com/repository/docker/gabcares/yassir-eta-shiny)

#### Demo video- shiny application

- Coming soon!

## Technologies Used

- Anaconda
- **Shiny for Python**
- Python
- Pandas
- Plotly
- Git
- Scipy
- Sklearn
- Adaboost
- Decision tree
- HistGradientBoost
- LinearRegression
- RandomForest
- GradientBoost
- **XGBoostRegressor**
- Joblib

## Installation

### Quick install

```bash
 pip install -r requirements.txt
```

### Recommended install

```bash
conda env create -f yassir-environment.yml
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 💻 Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

- [Docker Desktop](https://docs.docker.com/desktop/)

### Setup

Clone this repository to your desired folder:

```sh
  cd your-folder
  git clone https://github.com/D0nG4667/yassir_eta_prediction_shiny_app.git
```

Change into the cloned repository

```sh
  cd yassir_eta_prediction_shiny_app
  
```

After cloning this repo,

- Add an env folder in the root of the project.

- Create and env file named `offline.env` using this sample

```env
# API
API_URL=http://api:7860/api/v1/eta/prediction?model

# Google Maps API
MAPS_API_KEY=Your Key

# Redis local
REDIS_URL=redis://cache:6379/
REDIS_USERNAME=default
```

- Run these commands in the root of the repo to explore the frontend and backend application:

```sh
docker-compose pull

docker-compose build

docker-compose up

```

## Contributions

### How to Contribute

1. Fork the repository and clone it to your local machine.
2. Explore the Jupyter Notebooks and documentation.
3. Implement enhancements, fix bugs, or propose new features.
4. Submit a pull request with your changes, ensuring clear descriptions and documentation.
5. Participate in discussions, provide feedback, and collaborate with the community.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Feedback and Support

Feedback, suggestions, and contributions are welcome! Feel free to open an issue for bug reports, feature requests, or general inquiries. For additional support or questions, you can connect with me on [LinkedIn](https://www.linkedin.com/in/dr-gabriel-okundaye).


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 👥 Authors <a name="authors"></a>

🕺🏻**Gabriel Okundaye**

- GitHub: [GitHub Profile](https://github.com/D0nG4667)

- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/dr-gabriel-okundaye)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ⭐️ Show your support <a name="support"></a>

If you like this project kindly show some love, give it a 🌟 **STAR** 🌟. Thank you!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📝 License <a name="license"></a>

This project is [MIT](/LICENSE) licensed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

