# Install script for directory: /home/turtlepc/turtlebot_ws/src/turtlebot3_autorace_2020/turtlebot3_autorace_detect

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/turtlepc/turtlebot_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/safe_execute_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/turtlebot3_autorace_detect" TYPE FILE FILES "/home/turtlepc/turtlebot_ws/devel/include/turtlebot3_autorace_detect/DetectLaneParamsConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/turtlebot3_autorace_detect" TYPE FILE FILES "/home/turtlepc/turtlebot_ws/devel/include/turtlebot3_autorace_detect/DetectLevelParamsConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/turtlebot3_autorace_detect" TYPE FILE FILES "/home/turtlepc/turtlebot_ws/devel/include/turtlebot3_autorace_detect/DetectTrafficLightParamsConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/home/turtlepc/bin/python3" -m compileall "/home/turtlepc/turtlebot_ws/devel/lib/python3/dist-packages/turtlebot3_autorace_detect/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/turtlebot3_autorace_detect" TYPE DIRECTORY FILES "/home/turtlepc/turtlebot_ws/devel/lib/python3/dist-packages/turtlebot3_autorace_detect/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/turtlebot3_autorace_detect.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot3_autorace_detect/cmake" TYPE FILE FILES
    "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/turtlebot3_autorace_detectConfig.cmake"
    "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/turtlebot3_autorace_detectConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot3_autorace_detect" TYPE FILE FILES "/home/turtlepc/turtlebot_ws/src/turtlebot3_autorace_2020/turtlebot3_autorace_detect/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_construction")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_construction_sign")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_intersection")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_intersection_sign")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_lane")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_level_crossing")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_level_crossing_sign")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_parking")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_parking_sign")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_traffic_light")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_tunnel")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_autorace_detect" TYPE PROGRAM FILES "/home/turtlepc/turtlebot_ws/build/turtlebot3_autorace_2020/turtlebot3_autorace_detect/catkin_generated/installspace/detect_tunnel_sign")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot3_autorace_detect" TYPE DIRECTORY FILES
    "/home/turtlepc/turtlebot_ws/src/turtlebot3_autorace_2020/turtlebot3_autorace_detect/cfg"
    "/home/turtlepc/turtlebot_ws/src/turtlebot3_autorace_2020/turtlebot3_autorace_detect/image"
    "/home/turtlepc/turtlebot_ws/src/turtlebot3_autorace_2020/turtlebot3_autorace_detect/launch"
    "/home/turtlepc/turtlebot_ws/src/turtlebot3_autorace_2020/turtlebot3_autorace_detect/param"
    )
endif()

