# Import third party packages onto AWS Lambda

## 1. Build-in AWS layers
- Pandas and Numpy are already build-in

## 2. Zip third-party packages
### 2.1. Python layer compatibility with Amazon Linux

1. In Python, most packages are available as wheels (.whl files) in addition to the source distribution.
2. Wheels are useful for ensuring that your layer is compatible with Amazon Linux.
3. When you download your dependencies, download the universal wheel if possible. 
4. If not the universal one, then choose the "manylinux" with "x86_64" and corresponding Python version.

### 2.2. Layer paths for Python runtimes

When you add a layer to a function, Lambda loads the layer content into the /opt directory of that execution environment. For each Lambda runtime, the PATH variable already includes specific folder paths within the /opt directory. To ensure that the PATH variable picks up your layer content, your layer .zip file should have its dependencies in the following folder paths:
- python
- python/lib/python3.x/site-packages

For example, the resulting layer .zip file that you create in this tutorial has the following directory structure:
```bash
   layer_content.zip
   └ python
       └ lib
           └ python3.11
               └ site-packages
                   └ requests
                   └ <other_dependencies> (i.e. dependencies of the requests package)
                   └ ...
```

The requests library is correctly located in the python/lib/python3.11/site-packages directory. This ensures that Lambda can locate the library during function invocations.

### 2.3. Packaging the layer content

#### 2.3.1. Third-party package with universal wheel - i.e. requests with Python 3.11
- Example requirements.txt
```bash
  requests==2.31.0
```

This script uses venv to create a Python virtual environment named create_layer. It then installs all required dependencies in the create_layer/lib/python3.11/site-packages directory.
```bash
  python3.11 -m venv create_layer
  source create_layer/bin/activate
  pip install -r requirements.txt
```

This script copies the contents from the create_layer/lib directory into a new directory named python. It then zips the contents of the python directory into a file named layer_content.zip. This is the .zip file for your layer. You can unzip the file and verify that it contains the correct file structure, as shown in the Layer paths for Python runtimes section.
```bash
  mkdir python
  cp -r create_layer/lib python/
  zip -r layer_content.zip python
```

#### 2.3.2. Working with manylinux wheel distributions - i.e. numpy with Python 3.11
- Example requirements.txt
- Here, you specify the URL of the manylinux wheel distribution that's compatible with Python 3.11, Amazon Linux, and the x86_64 instruction set
```bash
  https://files.pythonhosted.org/packages/3a/d0/edc009c27b406c4f9cbc79274d6e46d634d139075492ad055e3d68445925/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

This script uses venv to create a Python virtual environment named create_layer. It then installs all required dependencies in the create_layer/lib/python3.11/site-packages directory. The pip command is different in this case, because you must specify the --platform tag as manylinux2014_x86_64. This tells pip to install the correct manylinux wheel, even if your local machine uses macOS or Windows.
```bash
  python3.11 -m venv create_layer
  source create_layer/bin/activate
  pip install -r requirements.txt --platform=manylinux2014_x86_64 --only-binary=:all: --target ./create_layer/lib/python3.11/site-packages
```

This script copies the contents from the create_layer/lib directory into a new directory named python. It then zips the contents of the python directory into a file named layer_content.zip. This is the .zip file for your layer. You can unzip the file and verify that it contains the correct file structure as shown in the Layer paths for Python runtimes section.
```bash
  mkdir python
  cp -r create_layer/lib python/
  zip -r layer_content.zip python
```

### 3. Add layers
1. Upload zip files onto Lambda layer
2. Have the Lambda function to add the layer

## Official Documents
1. [Working with .zip file archives for Python Lambda functions](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html)
2. [Working with layers for Python Lambda functions](https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html)
