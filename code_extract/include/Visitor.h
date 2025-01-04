#pragma once

#include "CodeDB.h"
#include "VisitManager.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTUnit.h"

namespace clang {
class ASTContext;
class CXXRecordDecl;
class VarDecl;
} // namespace clang

class CodeExtractVisitor
    : public clang::RecursiveASTVisitor<CodeExtractVisitor> {
  CodeDB &codedb;
  clang::ASTUnit &unit;
  clang::ASTContext &ctx;
  std::string const& projPath;

public:
  CodeExtractVisitor(CodeDB &db, clang::ASTUnit &astUnit, std::string const& projPath)
      : codedb(db), unit(astUnit), ctx(astUnit.getASTContext()), projPath(projPath) {}
  bool VisitVarDecl(clang::VarDecl *decl);
  bool VisitFunctionDecl(clang::FunctionDecl *decl);
  bool VisitRecordDecl(clang::RecordDecl *decl);
  bool VisitTypedefNameDecl(clang::TypedefNameDecl *decl);
};

class MatchVisitor : public clang::RecursiveASTVisitor<MatchVisitor> {
  VisitManager &vm;
  CodeDB const &codedb;

public:
  MatchVisitor(VisitManager &visitm, CodeDB const &db)
      : vm(visitm), codedb(db) {}
  bool VisitCallExpr(clang::CallExpr *decl);
  bool VisitDeclRefExpr(clang::DeclRefExpr *decl);
  bool VisitCXXConstructExpr(clang::CXXConstructExpr *expr);
  // Custom Defined
  void VisitParms(clang::FunctionDecl const *defDecl);
};