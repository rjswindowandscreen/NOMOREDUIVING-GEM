# How to contribute
This document describes how to contribute this project.  
Before starting your contribution, please check [README](/README.md) to understand what this project is and setup your development environment.  

## Add new models and worlds
Steps to add a new model or world.  

### Step 1: Propose by creating issue
If you found an algorithm which you want to implement, let's create an issue to propose it on GitHub. We will check it and give a comment as soon as possible.

If we agreed with your proposal, let's go to Step 2.  

It's OK to just create an issue to propose. Someone might see your proposal and implement in the future. In this case, please a paper or documentation about the algorithm to understand it.  

### Step 2: Implement sample program of algorithm
When you implement a sample program of an algorithm, please keep the following items in mind.  

1. Use only Python. Using other language is not acceptable.  
2. Acceptable version of Python is 3.x.  
3. Follow ros2 gz 

### Step 3: Submit a pull request and modify code based on review
If your sample program and test were ready, let's create a pull request and submit it. When you create the PR, please write a description about the following items.  

* The overview of your PR.  
* How did you confirm that your program works well.  

After you submitted your PR, each unit tests is executed automatically by GitHub Actions. If an error occured and unit tests failed, please investigate the reason and fix it. After all tests passed and any problems were not found by code review, we will approve you to merge your PR into main branch.  

## Report and fix defect
Reporting and fixing a defect are also welcome.  
When you report an issue, please provide the following information.  

* A clear and concise description about the defect.  
* A clear and consice description about your expectation.  
* Screenshots to help explaining the defect.
* OS version
* ROS version.  
* Python version.  


## Add documentation about existing program
Adding a documentation about existing programs is also welcome.  
There have not been any rules about documentation yet. If you had any suggestion of how to write documentations, please submit a PR and we will review it.