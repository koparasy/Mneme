#pragma once
#include <memory>
#include <vector>

#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"

#include "CodeDB.h"

namespace clang {
class ASTUnit;
class FunctionDecl;
} // namespace clang

/// @brief Exposes API to pull a single function by name and manages other tool
/// aspects.
class ToolManager {
  std::unique_ptr<clang::tooling::CompilationDatabase> compDb;
  std::unique_ptr<clang::tooling::ClangTool> tool;
  ObjInfo *primaryFn;
  std::vector<std::unique_ptr<clang::ASTUnit>> asts;
  std::unique_ptr<CodeDB> db;

  std::string const& projPath;

public:
  ToolManager(std::string const &projectDirPath);

  /// @brief Finds the function declaration by name.
  /// @param fnName Name of the function declaration to find.
  /// @return The function declaration if a function by that name exists in the
  /// tool's compilation database, null otherwise.
  ObjInfo *findFnDeclByName(std::string const &fnName);

  /// @brief Pulls the function definition for the specified function name and
  /// all its dependencies into file <fn-name>.<fn-source-extn> and compiles it
  /// to <fn-name>.o.
  /// @param fnName Name of the function to get dependencies for.
  void getStandaloneFnContext(std::string const &fnName);
};
