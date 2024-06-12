#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/turtlepc/turtlebot_ws/src/turtlebot3_autorace_2020/turtlebot3_autorace_core"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/turtlepc/turtlebot_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/turtlepc/turtlebot_ws/install/lib/python3/dist-packages:/home/turtlepc/turtlebot_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/turtlepc/turtlebot_ws/build" \
    "/home/turtlepc/bin/python3" \
    "/home/turtlepc/turtlebot_ws/src/turtlebot3_autorace_2020/turtlebot3_autorace_core/setup.py" \
     \
    build --build-base "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_core" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/turtlepc/turtlebot_ws/install" --install-scripts="/home/turtlepc/turtlebot_ws/install/bin"
