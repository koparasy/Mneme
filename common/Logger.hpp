#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

namespace mneme {

class Logger {
public:
  static std::ofstream &logs(const std::string &Name) {
    static Logger SingletonLogger{Name};
    return SingletonLogger.OutStream;
  }

private:
  const std::string LogDir = "mneme-logs";
  bool DirExists;
  std::error_code EC;
  std::ofstream OutStream;

  Logger(std::string Name)
      : DirExists(std::filesystem::create_directory(LogDir)),
        OutStream(std::ofstream{
            LogDir + "/" + Name + "." + std::to_string(getpid()) + ".log",
        }) {
    if (!OutStream.good())
      throw std::runtime_error("Error opening file: " + EC.message());
  }
};

} // namespace mneme
