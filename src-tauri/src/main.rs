fn main() {
    #[cfg(target_os = "linux")]
    {
        let args: Vec<String> = std::env::args().collect();
        if args.iter().any(|arg| arg == "--graphics=compatible")
            && std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none()
        {
            // Must be set before Tauri creates WebKit or starts its threads.
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
        if args.iter().any(|arg| arg == "--diagnose") {
            println!("ModelBuilder {}", env!("CARGO_PKG_VERSION"));
            println!("Arquitectura: {}", std::env::consts::ARCH);
            println!(
                "Sesión: {}",
                std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "desconocida".into())
            );
            println!(
                "WEBKIT_DISABLE_DMABUF_RENDERER: {}",
                std::env::var("WEBKIT_DISABLE_DMABUF_RENDERER")
                    .unwrap_or_else(|_| "no configurado".into())
            );
            println!("Prueba de compatibilidad: modelbuilder --graphics=compatible");
            return;
        }
    }
    modelbuilder_lib::run()
}
