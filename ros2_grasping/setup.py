from setuptools import find_packages, setup

package_name = 'ros2_grasping'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonardo',
    maintainer_email='leonardosabbatini12@gmail.com',
    description='ROS 2 wrapper for ICG-Net grasping model',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grasp_planner_node = ros2_grasping.grasp_planner_node:main'
        ],
    },
)
