# Annotation Tool

This folder contains scripts and the annotation tool for exercise 4.3. For exercise 4.2, check seg.py.

To get started, it is recommended to create a new environment. You can reuse your existing environment if you prefer though.
Install the new dependencies using `pip install -r requirements.txt`.

## Using the Annotation Tool

Before using the annotation tool, you have to complete exercise 4.2 (this tool depends on `seg.py`).

In order to use the tool, you must first create an `imgs` folder inside a data directory.
Put all you image files into the `imgs` folder. Ideally, you create a `data` directory in the same directory as `app.py`.

If you have activated a virtual environment using the required dependencies, you can run `python app.py` to launch the annotation tool.

The terminal will tell you to open a web browser on a specific location.
When you open the web page, it will tell you that no ?img was specified. Modify the URL and append something like `?img=12345678`.
Replace the number with one of the numbers of you files (don't add the file extension).

With every click, the result will be immediately saved to your data directory.

The annotation tool is a web server and has a fixed port assigned. If you run the tool on
one of the computers in our room, the default port might be already occupied by another student.
In that case, you have to specify the `--port` option and change the port to another one
(ports starting from 32000 are rarely used).

### SSH Port Forwarding

When you run the annotation tool on a remote machine, the web server is only available on the remote.
To make it available locally, you can use SSH port forwarding. This will open a tunnel from you local machine to the remote machine via SSH.
With an open SSH tunnel, it will appear as if the server on the remote machine runs on your local machine.
To start SSH forwarding, use the following command:

    ssh -L LOCAL_PORT:localhost:DESTINATION_PORT [USER@]SSH_SERVER

Replace `DESTINATION_PORT` with the port used on the remote machine and `LOCAL_PORT` with a port you would like to use locally (.e.g. 5000).
Next, open you web browser on `http://localhost:LOCAL_PORT/index.html` (replace `LOCAL_PORT`).

If you are using PyCharm, you can set up port forwarding as described here: https://www.jetbrains.com/help/pycharm/security-model.html#port_forwarding
