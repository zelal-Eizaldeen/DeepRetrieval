import os

def get_jdk_path():
    home = os.environ.get('HOME')
    jdk_dir = os.path.join(home, ".jdk")
    
    if not os.path.exists(jdk_dir):
        raise RuntimeError(f"JDK directory not found at {jdk_dir}")
    
    # List all directories in .jdk that start with 'jdk-'
    jdk_versions = [d for d in os.listdir(jdk_dir) if d.startswith('jdk-')]
    
    if not jdk_versions:
        raise RuntimeError(f"No JDK installations found in {jdk_dir}")
    
    # Get the latest version
    latest_version = sorted(jdk_versions)[-1]
    return os.path.join(home, ".jdk", latest_version)

os.environ['JAVA_HOME'] = get_jdk_path()
os.environ['PATH'] = os.environ['PATH'] + ':' + os.path.join(os.environ['JAVA_HOME'], 'bin')
