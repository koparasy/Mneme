#pragma once
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/ASTUnit.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace clang {
class Decl;
class ASTUnit;
} // namespace clang

/// @brief Stores uniquely identifying information for AST nodes of interest.
class ObjInfo {
  clang::ASTUnit const &unit;
  std::string const name;
  clang::Decl *decl;
  clang::Decl *def = nullptr;
  bool defInSameTU = false;
  // If this decl is external to the project, store its source file's name for include'ing later.
  std::string extSourceFile = "";

  clang::Decl *getDef() const { return def ? def : decl; }

public:
  ObjInfo(clang::ASTUnit const &astUnit, std::string name,
          clang::Decl *mainDecl, clang::Decl *defDecl = nullptr)
      : unit(astUnit), name(name), decl(mainDecl), def(defDecl),
        defInSameTU(defDecl) {}
  void addDefinitionDecl(clang::Decl *defDecl) { def = defDecl; }

  clang::Decl *getDefiniton() { return getDef(); }
  clang::Decl const *getDefiniton() const { return getDef(); }

  bool isDefInSameTU() const { return defInSameTU; }
  
  std::string getName() const { return name; }
  
  /// Get filename from which this specific decl is referenced.
  clang::StringRef getRefFile() const {
    return unit.getOriginalSourceFileName();
  }

  void addExtSourceFile(std::string file) { extSourceFile = file; }

  std::string getExtSourceFile() const { return extSourceFile;}
};

class CodeDB {
  std::unordered_map<std::string, std::unique_ptr<ObjInfo>> db;

  clang::Decl *getDef(std::string const &name) const {
    if (isRegistered(name))
      return db.at(name)->getDefiniton();
    else
      return nullptr;
  }

  ObjInfo *getObjInfo(std::string const &name) const {
    if (isRegistered(name))
      return db.at(name).get();
    else
      return nullptr;
  }

public:
  bool isRegistered(std::string const &name) const {
    return db.find(name) != db.end();
  }
  void registerDecl(clang::ASTUnit const &unit, std::string name,
                    clang::Decl *decl, clang::Decl *defDecl = nullptr) {
    db.try_emplace(name, std::make_unique<ObjInfo>(unit, name, decl, defDecl));
  }

  ObjInfo const *getObjInfoOrNull(std::string const &name) const {
    return getObjInfo(name);
  }

  ObjInfo *getObjInfoOrNull(std::string const &name) {
    return getObjInfo(name);
  }

  clang::Decl const *getDefinitionDecl(std::string const &name) const {
    return getDef(name);
  }

  clang::Decl *getDefinitionDecl(std::string const &name) {
    return getDef(name);
  }

  void addDefinitionDecl(std::string const &name, clang::Decl *defDecl) {
    if (isRegistered(name))
      db.at(name)->addDefinitionDecl(defDecl);
  }

  void addExtSource(std::string const& name, std::string const& fileName) {
    if (isRegistered(name))
      db.at(name)->addExtSourceFile(fileName);
  }

  std::string getExtSource(std::string const& name) const {
    if (db.find(name) == db.end()) return "";
    return db.at(name)->getExtSourceFile();
  }
};