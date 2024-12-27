#include "ToolManager.h"
#include "CodeDB.h"
#include "Visitor.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_set>

namespace ct = clang::tooling;

namespace helper {
std::string
getCompilationFlags(std::vector<std::string> const &cli,
                    std::unordered_set<std::string> const &blacklistedFlags) {
  std::string flags;
  for (auto &command : cli) {
    if (command[0] == '-' &&
        blacklistedFlags.find(command) == blacklistedFlags.end())
      flags += command + " ";
  }
  return flags;
}
} // namespace helper

ToolManager::ToolManager(std::string const &dirPath) {
  // Setup our tool
  std::string errorMsg;
  compDb = ct::CompilationDatabase::autoDetectFromDirectory(dirPath, errorMsg);
  if (!errorMsg.empty()) {
    std::cerr << errorMsg << std::endl;
    exit(1);
  }
  auto sourceFiles = compDb->getAllFiles();
  tool = std::make_unique<ct::ClangTool>(*compDb, sourceFiles);

  // Build code database
  tool->buildASTs(asts);
  db.reset(new CodeDB());
  // Build code database
  for (auto &ast : asts) {
    CodeExtractVisitor vis(*db.get(), *ast);
    // Should move this in a run function or ctor
    vis.TraverseAST(ast->getASTContext());
  }
}

ObjInfo *ToolManager::findFnDeclByName(std::string const &fnName) {
  auto obj = db->getObjInfoOrNull(fnName);

  if (!obj) {
    std::cerr << "Could not find function(s) named " << fnName
              << " within specified project!" << std::endl;
    exit(1);
  }
  if (!obj->getDefiniton()) {
    std::cerr << "Could not find body for function(s) named " << fnName
              << " within specified project! Make sure it is not an externally "
                 "defined function!"
              << std::endl;
    exit(1);
  }

  return obj;
}

void ToolManager::getStandaloneFnContext(std::string const &fnName) {
  primaryFn = findFnDeclByName(fnName);
  auto fnSrcFile = primaryFn->getSourceFile();
  bool isCuda = fnSrcFile.substr(fnSrcFile.size() - 2, 2) == "cu";
  std::string filename = fnName + (isCuda ? ".cu" : ".cpp");
  std::string objname = fnName + ".o";

  VisitManager mv(*primaryFn, *db.get());

  mv.pullPrimaryFnContext();
  std::string code;
  mv.emitStandaloneFile(code);

  // Compile pulled code
  // First dump everything to a file
  std::ofstream outFile(filename);

  if (!outFile) {
    std::cerr << "Error creating file " << filename << std::endl;
    return;
  }

  outFile << code;
  outFile.close();

  // Then clang-format
  std::string command = "clang-format -i " + filename;
  if (system(command.c_str()) != 0)
    std::cerr << "Formatting failed!" << std::endl;

  // Then compile
  // For now only get one compilation command
  auto cli = compDb->getCompileCommands(fnSrcFile)[0].CommandLine;
  command = cli[0] + " -o " + objname + " " + filename + " " +
            helper::getCompilationFlags(cli, {"-c", "-o", "--driver-mode=g++"});
  std::cout << "Compiling " << fnName << " with command:\n"
            << command << std::endl;
  if (system(command.c_str()) != 0) {
    std::cerr << "Compilation failed!" << std::endl;
  } else {
    std::cout << "Compilation successful!" << std::endl;
  }
  /// FIXME: We need to add a lot more here....should not be restrited to a
}