#include <filesystem>
#include <iostream>
#include <vector>

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"

#include "ToolManager.h"

namespace ct = clang::tooling;

int main(int argc, const char **argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: code-extract path/to/dir function-name\n"; // typically the
                                                          // compile commands
                                                          // will be in build.
    return 1;
  }
  std::string dirPath = argv[1];
  std::string fnName = argv[2]; // Possibly needs to be fully qualified...

  if (!std::filesystem::is_directory(dirPath)) {
    std::cerr << "Error: Directory '" << dirPath << "' not found.\n";
    return 1;
  }

  std::string errorMsg;
  // set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE), need the compilation database to
  // recall all compile commands and source files.
  auto compDatabase =
      ct::CompilationDatabase::autoDetectFromDirectory(dirPath, errorMsg);
  if (!errorMsg.empty()) {
    std::cerr << errorMsg << std::endl;
    return 1;
  }
  auto sourceFiles = compDatabase->getAllFiles();

  // Setup our tool and manager
  ct::ClangTool tool(*compDatabase, sourceFiles);
  ToolManager tm(&tool);

  // For not this function does all the work but we should probably make a
  // cleaner interface...
  tm.getStandaloneFnContext(fnName);
}