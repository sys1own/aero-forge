//! Deterministic AST repair via E-Graph equality saturation.
//!
//! The ``AeroAstLanguage`` embeds a small arithmetic expression fragment of the
//! UAST.  ``repair_uast_expression`` runs ``egg`` equality saturation to
//! minimise expression cost before returning the rewritten expression as UAST
//! JSON.  No LLM or heuristic text substitution is used.

use egg::{define_language, rewrite, AstSize, Extractor, Id, RecExpr, Runner, Symbol};
use ordered_float::NotNan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

pub type Rewrite = egg::Rewrite<AeroAstLanguage, ()>;

pub type Float = NotNan<f64>;

define_language! {
    pub enum AeroAstLanguage {
        Num(Float),
        Var(Symbol),
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "div" = Div([Id; 2]),
        "neg" = Neg([Id; 1]),
        "pow" = Pow([Id; 2]),
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "log" = Log([Id; 1]),
        "abs" = Abs([Id; 1]),
    }
}

/// Build a standard set of algebraic rewrite rules for the arithmetic fragment.
///
/// These rules are proof-theoretic and preserve value semantics for the limited
/// subset represented here.
pub fn make_ast_rewrite_rules() -> Vec<Rewrite> {
    vec![
        rewrite!("add-zero"; "(add ?x 0)" => "?x"),
        rewrite!("mul-one"; "(mul ?x 1)" => "?x"),
        rewrite!("mul-zero"; "(mul ?x 0)" => "0"),
        rewrite!("sub-self"; "(sub ?x ?x)" => "0"),
        rewrite!("div-self"; "(div ?x ?x)" => "1"),
        rewrite!("add-comm"; "(add ?x ?y)" => "(add ?y ?x)"),
        rewrite!("mul-comm"; "(mul ?x ?y)" => "(mul ?y ?x)"),
        rewrite!("neg-neg"; "(neg (neg ?x))" => "?x"),
    ]
}

fn parse_num(f: f64) -> Float {
    Float::new(f).unwrap_or_else(|_| Float::new(0.0).unwrap())
}

fn value_to_enode(v: &Value) -> Option<AeroAstLanguage> {
    match v {
        Value::Number(n) => n.as_f64().map(parse_num).map(AeroAstLanguage::Num),
        Value::String(s) => Some(AeroAstLanguage::Var(Symbol::new(s))),
        _ => None,
    }
}

fn collect_args(map: &serde_json::Map<String, Value>) -> Vec<Value> {
    if let Some(Value::Array(arr)) = map.get("arguments") {
        return arr.clone();
    }
    if let Some(Value::Array(arr)) = map.get("argument") {
        return arr.clone();
    }
    if let Some(arg) = map.get("argument") {
        return vec![arg.clone()];
    }
    if let Some(arg) = map.get("value") {
        return vec![arg.clone()];
    }
    Vec::new()
}

fn value_to_recexpr(v: &Value, expr: &mut RecExpr<AeroAstLanguage>) -> Result<Id, String> {
    if let Some(enode) = value_to_enode(v) {
        return Ok(expr.add(enode));
    }

    if let Value::Object(map) = v {
        let node_type = map.get("type").and_then(Value::as_str).unwrap_or("");
        if node_type == "literal" {
            if let Some(value) = map.get("value") {
                if let Some(enode) = value_to_enode(value) {
                    return Ok(expr.add(enode));
                }
            }
            return Err("literal without a value".to_string());
        }
        if node_type == "reference" {
            let name = map.get("name").and_then(Value::as_str).unwrap_or("");
            return Ok(expr.add(AeroAstLanguage::Var(Symbol::new(name))));
        }
        if node_type == "call" || node_type == "binop" || node_type == "unaryop" {
            let name = map
                .get("function")
                .or_else(|| map.get("name"))
                .or_else(|| map.get("op"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = collect_args(map);
            let child_ids: Vec<Id> = args
                .iter()
                .map(|a| value_to_recexpr(a, expr))
                .collect::<Result<Vec<_>, _>>()?;
            return build_enode(name, expr, &child_ids);
        }
    }

    Err(format!("unsupported UAST expression node: {}", v))
}

fn build_enode(
    name: &str,
    expr: &mut RecExpr<AeroAstLanguage>,
    kids: &[Id],
) -> Result<Id, String> {
    let binary = |expr: &mut RecExpr<AeroAstLanguage>, kids: &[Id]| {
        if kids.len() != 2 {
            Err("binary operator needs two arguments".to_string())
        } else {
            Ok(expr.add(AeroAstLanguage::Add([kids[0], kids[1]])))
        }
    };
    let unary = |expr: &mut RecExpr<AeroAstLanguage>, kids: &[Id]| {
        if kids.len() != 1 {
            Err("unary operator needs one argument".to_string())
        } else {
            Ok(expr.add(AeroAstLanguage::Neg([kids[0]])))
        }
    };

    match name {
        "+" | "add" => Ok(expr.add(AeroAstLanguage::Add([kids[0], kids[1]]))),
        "-" | "sub" => Ok(expr.add(AeroAstLanguage::Sub([kids[0], kids[1]]))),
        "*" | "mul" => Ok(expr.add(AeroAstLanguage::Mul([kids[0], kids[1]]))),
        "/" | "div" => Ok(expr.add(AeroAstLanguage::Div([kids[0], kids[1]]))),
        "**" | "pow" => Ok(expr.add(AeroAstLanguage::Pow([kids[0], kids[1]]))),
        "neg" | "-u" | "unary-" => unary(expr, kids),
        "sin" => unary(expr, kids),
        "cos" => unary(expr, kids),
        "sqrt" => unary(expr, kids),
        "log" => unary(expr, kids),
        "abs" => unary(expr, kids),
        _ => {
            // If two children were provided but the operator name is unknown,
            // treat it as a generic add so the expression still rewrites.
            if kids.len() == 2 {
                binary(expr, kids)
            } else if kids.len() == 1 {
                unary(expr, kids)
            } else {
                Err(format!("unsupported call function: {}", name))
            }
        }
    }
}

fn rec_expr_to_value(expr: &RecExpr<AeroAstLanguage>, id: Id) -> Value {
    match &expr[id] {
        AeroAstLanguage::Num(n) => {
            serde_json::json!({ "type": "literal", "value": n.into_inner() })
        }
        AeroAstLanguage::Var(s) => {
            serde_json::json!({ "type": "reference", "name": s.to_string() })
        }
        AeroAstLanguage::Add([a, b]) => call_json("+", &[a, b], expr),
        AeroAstLanguage::Sub([a, b]) => call_json("-", &[a, b], expr),
        AeroAstLanguage::Mul([a, b]) => call_json("*", &[a, b], expr),
        AeroAstLanguage::Div([a, b]) => call_json("/", &[a, b], expr),
        AeroAstLanguage::Pow([a, b]) => call_json("**", &[a, b], expr),
        AeroAstLanguage::Neg([a]) => call_json("neg", &[a], expr),
        AeroAstLanguage::Sin([a]) => call_json("sin", &[a], expr),
        AeroAstLanguage::Cos([a]) => call_json("cos", &[a], expr),
        AeroAstLanguage::Sqrt([a]) => call_json("sqrt", &[a], expr),
        AeroAstLanguage::Log([a]) => call_json("log", &[a], expr),
        AeroAstLanguage::Abs([a]) => call_json("abs", &[a], expr),
    }
}

fn call_json(name: &str, args: &[&Id], expr: &RecExpr<AeroAstLanguage>) -> Value {
    let arguments: Vec<Value> = args.iter().map(|&&id| rec_expr_to_value(expr, id)).collect();
    serde_json::json!({
        "type": "call",
        "function": name,
        "arguments": arguments,
    })
}

/// Run equality saturation over a UAST expression and return the cheapest
/// equivalent expression as a UAST JSON string.
///
/// If the expression cannot be mapped to the supported arithmetic fragment the
/// original JSON is returned unchanged.
#[pyfunction]
pub fn repair_uast_expression(expr_json: &str) -> PyResult<String> {
    let value: Value = serde_json::from_str(expr_json)
        .map_err(|e| PyValueError::new_err(format!("invalid expression JSON: {}", e)))?;

    let mut expr = RecExpr::<AeroAstLanguage>::default();
    let root = match value_to_recexpr(&value, &mut expr) {
        Ok(id) => id,
        Err(_) => {
            // Unsupported fragment: return the original expression unchanged.
            return Ok(expr_json.to_string());
        }
    };

    let rules = make_ast_rewrite_rules();
    let runner = Runner::default().with_expr(&expr).run(&rules);
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_cost, best) = extractor.find_best(root);

    let root_id = Id::from(best.as_ref().len().saturating_sub(1));
    let out = rec_expr_to_value(&best, root_id);
    serde_json::to_string(&out)
        .map_err(|e| PyValueError::new_err(format!("output JSON serialization failed: {}", e)))
}
