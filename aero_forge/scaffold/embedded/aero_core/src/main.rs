//! Minimal CLI entrypoint for standalone `.aeroc` wavefront execution.

use aero_core::wavefront::{execute, Task};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "--version" {
        println!("aeroc-runner {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // Default demo pipeline: echo two independent tasks then a join task.
    let tasks = vec![
        Task::new("setup", "echo 'aeroc setup'"),
        Task::new("compile", "echo 'aeroc compile'"),
        Task::new("verify", "echo 'aeroc verify'")
            .with_deps(&["setup", "compile"]),
    ];

    match execute(&tasks, None, Some(30)) {
        Ok(results) => {
            for r in &results {
                if !r.stdout.is_empty() {
                    print!("{}", r.stdout);
                }
                if !r.stderr.is_empty() {
                    eprint!("{}", r.stderr);
                }
                if r.exit_code != 0 {
                    std::process::exit(r.exit_code);
                }
            }
            println!("[WAVE] aeroc pipeline completed");
        }
        Err(e) => {
            eprintln!("[WAVE] error: {}", e);
            std::process::exit(1);
        }
    }
}
