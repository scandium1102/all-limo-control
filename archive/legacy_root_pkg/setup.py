from setuptools import setup, find_packages
setup(
    name="limo_control",
    version="0.1.0",
    packages=find_packages(),   # 自動包含 patrol_modules
    install_requires=['rospy'],
)

