# NaviQAte: Functionality-guided Web Application Exploration

This repository includes the implementation of NaviQAte, a web application navigation tool that frames web application exploration as a question-and-answer task, generating action sequences for tasks and functionalities without requiring detailed parameters.


## Instructions

### Step1: Environment setup
```
conda create -n naviqate --file requirements.txt
```

### Step 2: Activate the environment
```
conda activate naviqate
```

### Step 3: Add OpenAI credentials
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=<your_API_key>
```

### Step4: Run the evaluation
The evaluation script is included in the `evaluation` directory. First, change the directory to the `evaluation` directory using the following command:
```
cd evaluation
```
Then, run the evaluation bash script via:
```
./evaluate.sh
```
The results are stored in the `evaluation/out/concrete` directory.