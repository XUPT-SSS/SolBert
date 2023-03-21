import hashlib

from antlr4 import *

from solidity1_antlr4.SolidityLexer import SolidityLexer
from solidity1_antlr4.SolidityParser import SolidityParser as solidityparser

STATEMENT_ENDING_STRINGS = ["ContractDefinitionContext", "ContractPartContext", "StructDefinitionContext",
                            "ModifierDefinitionContext", "FunctionDefinitionContext", "EventDefinitionContext",
                            "EnumDefinitionContext", "MappingContext",
                            "BlockContext",
                            "IfStatementContext", "TryStatementContext", "WhileStatementContext", "ForStatementContext",
                            "InlineAssemblyStatementContext",
                            "DoWhileStatementContext", "ContinueStatementContext", "BreakStatementContext",
                            "ReturnStatementContext", "ThrowStatementContext", "EmitStatementContext",
                            "SimpleStatementContext", "UncheckedStatementContext", "RevertStatementContext"]


def normalize(node):
    # 字符串处理
    ty = node.getSymbol().type
    # print("type:{},text:{}".format(ty,node.getText()))
    if ty == 129:
        return str("stringliteral")
    # decimalNumber 类型处理
    elif ty == 104:
        ""
        return str("decimalnumber")
    # number_literal

    # hexNumber进制处理
    elif ty == 105:

        ""
        if int(node.getText(), 16) > 10000:
            return str("hexnumber")
        else:
            return node.getText()
    # Hexstring_liter
    elif ty == 106:
        return "numberunit"
    elif ty == 107:
        ""
        return str("hexliteral")
    ##标识符处理
    elif ty == 128:
        ""
        if len(node.getText()) == 1:
            return "simpleidentifier"
        # word = split_identifier(node.getText())
        return node.getText().lower()

    else:
        return node.getText()


def is_statement_node(node):
    """
    Return whether the node is a statement level node.

    Args:
        node (tree_sitter.Node): Node to be queried
        lang (str): Source code language

    Returns:
        bool: True if given node is a statement node

    """
    endings = STATEMENT_ENDING_STRINGS
    end = type(node).__name__
    end = str(end)
    if end in endings:
        return True
    else:
        return False


def __statement_xsbt(node):
    xsbt = []
    # if node is Endpoint get text
    if isinstance(node, TerminalNode):
        # normalize 规范化
        if node.getSymbol().type != -1:
            token = normalize(node)
            xsbt.append(token)

    else:
        # if node type is statement by my defined  add start_type
        if is_statement_node(node):
            xsbt.append("start_{}".format(type(node).__name__).replace("Context", '').lower())
        for i in range(0, node.getChildCount()):
            child = node.getChild(i)
            xsbt += __statement_xsbt(node=child)
            # add end_type
        if is_statement_node(node):
            xsbt.append("end_{}".format(type(node).__name__).replace("Context", '').lower())

    return xsbt


def getParser(path):
    try:
        input_stream = FileStream(path, encoding="utf-8")
        lexer = SolidityLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = solidityparser(stream)
        tree = parser.sourceUnit()
        result = []
        for i in range(0, tree.getChildCount()):
            child = tree.getChild(i)
            if isinstance(child, solidityparser.ContractDefinitionContext):
                res = __statement_xsbt(child)
                res = " ".join(res)
                result.append(res)
        return result
    except Exception:
        print("error_{}".format(path))


def getText(path):
    with open(path, 'r') as file:
        data = file.readlines()
        return data


def deduplicate(path,out_path):
    unique_contracts_code = set()
    lines = getText(path)
    f = open(out_path, "w")
    for line in lines:
        md5cal = hashlib.md5()
        md5cal.update(line.encode('utf-8'))
        md5val = md5cal.hexdigest()
        if md5val in unique_contracts_code:
            continue
        unique_contracts_code.add(md5val)
        line = line.strip()
        f.write(line + "\n")
