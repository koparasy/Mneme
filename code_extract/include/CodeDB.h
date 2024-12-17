#pragma once
#include <unordered_map>

namespace clang {
class Decl;
class ASTUnit;
} // namespace clang

/// @brief Stores uniquely identifying information for AST nodes of interest.
class ObjInfo {
  std::string const &name;
  clang::Decl *decl;
  clang::Decl *def = nullptr;
  bool defInSameTU = false;

  clang::Decl *getDef() const { return def ? def : decl; }

public:
  ObjInfo(std::string const &name, clang::Decl *mainDecl,
          clang::Decl *defDecl = nullptr)
      : name(name), decl(mainDecl), def(defDecl), defInSameTU(defDecl) {}
  void addDefinitionDecl(clang::Decl *defDecl) { def = defDecl; }

  clang::Decl *getDefiniton() { return getDef(); }
  clang::Decl const *getDefiniton() const { return getDef(); }

  bool isDefInSameTU() const { return defInSameTU; }
  std::string getName() const { return name; }
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
  void registerDecl(std::string name, clang::Decl *decl,
                    clang::Decl *defDecl = nullptr) {
    db.try_emplace(name, std::make_unique<ObjInfo>(name, decl, defDecl));
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
};