#pragma once

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CodeDB.h"
#include "VisitManager.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
class ASTContext;
class CXXRecordDecl;
class VarDecl;
} // namespace clang

class CodeExtractVisitor
    : public clang::RecursiveASTVisitor<CodeExtractVisitor> {
  CodeDB &codedb;
  clang::ASTContext &ctx;

public:
  CodeExtractVisitor(CodeDB &db, clang::ASTContext &astCtx)
      : codedb(db), ctx(astCtx) {}
  bool VisitVarDecl(clang::VarDecl *decl);
  bool VisitFunctionDecl(clang::FunctionDecl *decl);
  bool VisitRecordDecl(clang::RecordDecl *decl);
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
  void VisitParms(clang::FunctionDecl const* defDecl);
};