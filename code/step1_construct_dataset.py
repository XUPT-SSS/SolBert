import os
from antlr4 import *
import re
import utils
from solidity_parser.solidity_antlr4.SolidityLexer import SolidityLexer
import config as cf
from solidity1_antlr4.SolidityParser import SolidityParser as solidityparser

STATEMENT_ENDING_STRINGS=["ContractDefinitionContext","ContractPartContext","StructDefinitionContext","ModifierDefinitionContext","FunctionDefinitionContext","EventDefinitionContext","EnumDefinitionContext","MappingContext",
                          "BlockContext",
                          "IfStatementContext","TryStatementContext","WhileStatementContext","ForStatementContext","InlineAssemblyStatementContext",
                          "DoWhileStatementContext","ContinueStatementContext","BreakStatementContext","ReturnStatementContext","ThrowStatementContext","EmitStatementContext","SimpleStatementContext","UncheckedStatementContext","RevertStatementContext"]

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
        if int(node.getText(),16)> 10000:
            return str("hexnumber")
        else:
            return node.getText()
    #Hexstring_liter
    elif ty ==106:
        return "numberunit"
    elif ty ==107:
        ""
        return str("hexliteral")
    ##标识符处理
    elif ty==128:
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
def camel_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def split_identifier(identifier):
    """
    Split identifier into a list of subtokens.
    Tokens except characters and digits will be eliminated.

    Args:
        identifier (str): given identifier

    Returns:
        list[str]: list of subtokens
    """
    words = []

    word = re.sub(r'[^a-zA-Z0-9]', ' ', identifier)
    word = re.sub(r'(\d+)', r' \1 ', word)
    split_words = word.strip().split()
    for split_word in split_words:
        camel_words = camel_split(split_word)
        for camel_word in camel_words:
            words.append(camel_word.lower())

    return words
def  __statement_xsbt(node):
    """
    Method used to generate X-SBT recursively.

    Args:
        node (tree_sitter.Node): Root node to traversal
        lang (str): Source code language

    Returns:
        list[str]: List of strings representing node types

    """
    xsbt = []
    # if node is Endpoint get text
    if isinstance(node,TerminalNode):
            #normalize 规范化
            if node.getSymbol().type!=-1:
                token = normalize(node)
                xsbt.append(token)

    else:
        # if node type is statement by my defined  add start_type
        if is_statement_node(node):
            xsbt.append("start_{}".format(type(node).__name__).replace("Context",'').lower())
        for  i in range(0,node.getChildCount()):
            child = node.getChild(i)
            xsbt += __statement_xsbt(node=child)
                # add end_type
        if is_statement_node(node):
            xsbt.append("end_{}".format(type(node).__name__).replace("Context",'').lower())

    return xsbt

def inOrderTraversal(node, temp):
    if is_statement_node(node):
        temp.append("start_{}".format(type(node).__name__))
    for i in range(0,node.getChildCount()):
        child = node.getChild(i)
        if isinstance(child, TerminalNode):
            temp.append(child.getText())
        else:

            inOrderTraversal(child,temp)
    if is_statement_node(node):
        temp.append("end_{}".format(type(node).__name__))
def get_extractContract(node):
    result =[]
    for i in range(0, node.getChildCount()):
        child = node.getChild(i)
        res = []
        if isinstance(child,solidityparser.ContractDefinitionContext):
            for j in range(0,child.getChildCount()):
                temp = child.getChild(j)
                if isinstance(temp,solidityparser.ContractPartContext):
                    inOrderTraversal(temp,res)
                else:
                    if isinstance(temp,TerminalNode):
                        res.append(temp.getText())
                    else:
                        inOrderTraversal(temp,res)
        if len(res)==0:
            continue
        result.append(res)
    return result
def getParser(path):
    input_stream = FileStream(path,encoding="utf-8")
    lexer = SolidityLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = solidityparser(stream)
    # visitor = solidityvisitor()
    # context = parser.sourceUnit()
    return parser
def get_all_file_path_recur(path):
    None
def get_all_file_path(root):
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    result = []
    for file in files:
        result.append(os.path.join(root,file))
    return result
if __name__ == '__main__':
    path = cf.original_data_path
    files = get_all_file_path(path)
    targetPath=cf.target_data_path
    f = open(targetPath,"w")
    print(len(files))

    for file in files:
        parser = getParser(file)
        tree = parser.sourceUnit()
        # res = get_extractContract(tree)
        # print(res)
        t = 0
        for i in range(0,tree.getChildCount()):
            child = tree.getChild(i)
            if isinstance(child,solidityparser.ContractDefinitionContext):
                res = __statement_xsbt(child)
                res = " ".join(res)
                t = t+1
                print(res)
                f.write(res + "\n")
        print(t)
    f.close()
    utils.deduplicate(cf.target_data_path,cf.unique_data_path)


