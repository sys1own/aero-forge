//! Zero-dependency embedded wavefront execution engine for standalone exports.

pub mod goi_solver;
pub mod wavefront;

pub use goi_solver::{compute_metamorphic_gradients, execute_goi_wave, Matrix};
pub use wavefront::{compute_wavefronts, execute, ScheduleError, Task, TaskResult};

/// Simple facade used by generated `.aeroc` binaries.
pub struct WavefrontEngine {
    pub tasks: Vec<Task>,
    pub cwd: Option<String>,
    pub timeout_seconds: Option<u64>,
}

impl WavefrontEngine {
    pub fn new(cwd: Option<String>, timeout_seconds: Option<u64>) -> Self {
        Self {
            tasks: Vec::new(),
            cwd,
            timeout_seconds,
        }
    }

    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }

    pub fn run(&self) -> Result<Vec<TaskResult>, ScheduleError> {
        execute(&self.tasks, self.cwd.as_deref(), self.timeout_seconds)
    }
}
