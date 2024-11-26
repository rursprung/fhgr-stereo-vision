For this project part, the nlohmann/json for C++ is necessary to install. Find the folder 'nlohman' here in this subproject or install (globally) via json-develop.zip, downloadable from https://github.com/nlohmann/json


##--Using the Library in JetBrains CLion--
CLion uses CMake as its build system. Ensure the CMakeLists.txt file includes the correct paths for the header file json.hpp (or the nlohmann folder).

##--Using the Library in QtCreator--
QtCreator can also work with CMake or QMake as a build system. Here’s how to use the library with both:

###(a) If You're Using CMake in QtCreator
Same procedure as for CLion.

###(b) If You're Using QMake
Copy the json.hpp file into your project directory (e.g., include/nlohmann/json.hpp).
Add the include directory to your .pro file:
INCLUDEPATH += $$PWD/include

Use the library in your code as usual:
#include <nlohmann/json.hpp>
using json = nlohmann::json;

