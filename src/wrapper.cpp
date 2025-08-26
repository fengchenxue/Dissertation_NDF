
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>

static std::string g_path_distrib_to_surface;
static std::string g_path_virtual_gonio;
static std::string g_path_plot_scattering;
static bool g_d2s_split_stats = false;

// ---------- constants (avoid M_PI in MSVC) ----------
static constexpr float kPI = 3.14159265358979323846f;
static constexpr float kTwoPI = 6.28318530717958647692f;
static constexpr float kHalfPI = 1.57079632679489661923f;

// ---------- stb_image (HDR only, header-only) ----------
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_HDR
#include "stb_image.h"

// toggle to 1 if later you wire real upstream G1 here
#ifndef NDF_USE_UPSTREAM_G1
#define NDF_USE_UPSTREAM_G1 0
#endif

// ---------- tiny math ----------
struct Vec3f { float x, y, z; };
static inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
static inline float safe_acos(float x) { return std::acos(std::max(-1.f, std::min(1.f, x))); }



// ---------- shell helpers ----------
static std::string run_cmd(const std::string& cmd) {
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::ostringstream oss; oss << "Command failed (" << rc << "): " << cmd;
        throw std::runtime_error(oss.str());
    }
    return cmd;
}
static void run_cmd_log(const std::string& cmd, const std::string& log_path,
    const std::string& hint_when_fail = "")
{
    std::filesystem::create_directories(std::filesystem::path(log_path).parent_path());
    std::string full = cmd + " > \"" + log_path + "\" 2>&1";
    int rc = std::system(full.c_str());
    if (rc != 0) {
        std::ifstream lf(log_path);
        std::string head, line; int cnt = 0;
        while (std::getline(lf, line) && cnt < 50) { head += line + "\n"; ++cnt; }

        std::string dist_preview;
        {
            std::ifstream df("vg_tmp/dist.txt");
            std::string l; int c = 0;
            while (std::getline(df, l) && c < 50) { dist_preview += l + "\n"; ++c; }
        }

        std::ostringstream oss;
        oss << "Command failed (" << rc << "): " << cmd << "\n"
            << "---- LOG: " << log_path << " ----\n" << head;
        if (!dist_preview.empty()) {
            oss << "---- dist.txt preview ----\n" << dist_preview;
        }
        if (!hint_when_fail.empty()) oss << "HINT: " << hint_when_fail << "\n";
        throw std::runtime_error(oss.str());
    }
}

static std::string run_and_capture(const std::string& cmd, int& rc_out) {
    std::filesystem::create_directories("vg_tmp");
    const std::string tmp = "vg_tmp/_probe.txt";
    std::string full = cmd + " > \"" + tmp + "\" 2>&1";
    rc_out = std::system(full.c_str());
    std::ifstream f(tmp);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}


static void ensure_dir(const std::string& path) {
#ifdef _WIN32
    run_cmd("if not exist \"" + path + "\" mkdir \"" + path + "\"");
#else
    run_cmd("mkdir -p \"" + path + "\"");
#endif
}

// ---------- write inputs for upstream tools ----------
static std::string write_piecewise_txt(const std::vector<float>& Dx,
    const std::vector<float>& Dy,
    const std::string& out_txt)
{
    const int n = (int)Dx.size();
    if ((int)Dy.size() != n) throw std::runtime_error("Dx/Dy size mismatch");
    std::ofstream f(out_txt);
    if (!f) throw std::runtime_error("Cannot open " + out_txt);
    f << "# piecewise_linear_ndf\n";
    f << "n " << n << "\n";
    f << "Dx";
    for (int i = 0; i < n; ++i) f << " " << Dx[i];
    f << "\nDy";
    for (int i = 0; i < n; ++i) f << " " << Dy[i];
    f << "\n";
    return out_txt;
}

static void write_min_conductor(const std::string& path) {
    std::ofstream mf(path);
    if (!mf) throw std::runtime_error("Cannot open " + path);
    mf << "# simple conductor\n";
    mf << "type conductor\n";
    mf << "eta  10 10 10\n";
    mf << "k    5  5  5\n";
}

// ---------- HDR load (Radiance) ----------
static bool load_hdr_L(const std::string& filename, int& H, int& W, std::vector<float>& outL) {
    int w = 0, h = 0, n_in = 0;
    float* data = stbi_loadf(filename.c_str(), &w, &h, &n_in, 1);
    if (!data) return false;
    outL.assign(w * h, 0.0f);
    std::memcpy(outL.data(), data, sizeof(float) * w * h);
    stbi_image_free(data);
    H = h; W = w;
    return true;
}

// bilinear with horizontal wrap
static float bilinear_sample_wrap(const std::vector<float>& img, int Himg, int Wimg, float yimg, float ximg) {
    int y0 = (int)std::floor(yimg);
    int x0 = (int)std::floor(ximg);
    float ty = yimg - y0;
    float tx = ximg - x0;
    auto wrap = [&](int x) { int r = x % Wimg; return (r < 0) ? r + Wimg : r; };
    auto clamp = [&](int y) { return std::clamp(y, 0, Himg - 1); };
    int y1 = clamp(y0 + 1);
    int x1 = wrap(x0 + 1);
    y0 = clamp(y0);
    x0 = wrap(x0);

    const float v00 = img[y0 * Wimg + x0];
    const float v10 = img[y0 * Wimg + x1];
    const float v01 = img[y1 * Wimg + x0];
    const float v11 = img[y1 * Wimg + x1];
    const float v0 = v00 * (1.f - tx) + v10 * tx;
    const float v1 = v01 * (1.f - tx) + v11 * tx;
    return std::max(0.f, v0 * (1.f - ty) + v1 * ty);
}

// resample upstream theta-phi image to our (H,W) grid
static py::array_t<float> resample_theta_phi(const std::vector<float>& img, int H, int W, int Himg, int Wimg) {
    py::array_t<float> out({ H, W });
    auto o = out.mutable_unchecked<2>();
    for (int iy = 0; iy < H; ++iy) {
        const float theta = (iy + 0.5f) * (kHalfPI / float(H));
        const float yimg = theta / kHalfPI * float(Himg) - 0.5f;
        for (int ix = 0; ix < W; ++ix) {
            const float phi = (ix + 0.5f) * (kTwoPI / float(W));
            const float ximg = phi / kTwoPI * float(Wimg) - 0.5f;
            o(iy, ix) = bilinear_sample_wrap(img, Himg, Wimg, yimg, ximg);
        }
    }
    return out;
}

static void probe_d2s_stats_flag() {
    int rc = 0;
    std::string out = run_and_capture("\"" + g_path_distrib_to_surface + "\"", rc); // no args -> usage
    // Newer manual shows "--statistics <DIR>"
    // Older build prints "statistics-dir" and a boolean "statistics"
    if (out.find("statistics-dir") != std::string::npos) {
        g_d2s_split_stats = true;
    }
    else if (out.find("--statistics <") != std::string::npos) {
        g_d2s_split_stats = false;
    }
    else {
        // default to manual style for safety
        g_d2s_split_stats = false;
    }
}

// run upstream pipeline and produce HDRs
static std::string run_pipeline_hdr(const std::vector<float>& Dx,
    const std::vector<float>& Dy,
    float cos_ti, float sin_ti, float cos_pi, float sin_pi,
    int H, int W)
{
    const std::string work = "vg_tmp";
    ensure_dir(work);

    const std::string distrib_txt = work + "/dist.txt";
    write_piecewise_txt(Dx, Dy, distrib_txt);

    const std::string surf_obj = work + "/surf.obj";
    {
        std::filesystem::create_directories("vg_tmp");
        std::filesystem::create_directories("vg_tmp/stats");
        const std::string distrib_txt_abs = std::filesystem::absolute("vg_tmp/dist.txt").string();
        const std::string surf_obj_abs = std::filesystem::absolute("vg_tmp/surf.obj").string();
        const std::string stats_dir_abs = std::filesystem::absolute("vg_tmp/stats").string();

        std::ostringstream cmd;
        cmd << "\"" << g_path_distrib_to_surface << "\""
            << " --filename \"" << distrib_txt_abs << "\""
            << " --output \"" << surf_obj_abs << "\""
            << " --patch-size " << 1024
            << " --iterations " << 500;

        if (g_d2s_split_stats) {
            // older/alternate build: boolean + separate dir
            cmd << " --statistics"
                << " --statistics-dir \"" << stats_dir_abs << "\"";
        }
        else {
            // manual (current) build: single string arg
            cmd << " --statistics \"" << stats_dir_abs << "\"";
        }

        run_cmd_log(cmd.str(), "vg_tmp/d2s.log",
            "distrib_to_surface failed. Check d2s.log and dist.txt format.");
    }


    write_min_conductor(work + "/material_conductor.txt");

    const double theta_i = safe_acos(cos_ti);
    const double phi_i = std::atan2(sin_pi, cos_pi);
    const std::string out_base = work + "/bsdf";
    {
        const std::string distrib_txt_abs = std::filesystem::absolute("vg_tmp/dist.txt").string();
        const std::string surf_obj_abs = std::filesystem::absolute("vg_tmp/surf.obj").string();
        const std::string material_abs = std::filesystem::absolute(work + "/material_conductor.txt").string();
        const std::string out_base_abs = std::filesystem::absolute(out_base).string();

        std::ostringstream cmd;
        cmd << "\"" << g_path_virtual_gonio << "\""
            << " --filename \"" << distrib_txt_abs << "\""
            << " --surface \"" << surf_obj_abs << "\""
            << " --material \"" << material_abs << "\""
            << " --output \"" << out_base_abs << "\""
            << " --theta-i " << theta_i
            << " --phi-i " << phi_i
            << " --samples " << 128;

        run_cmd_log(cmd.str(), "vg_tmp/gonio.log");
    }



    const int m = std::max(H, W);
    const int image_size = (m % 2 == 0) ? m : (m + 1); // even
    {
        const std::string out_base_abs = std::filesystem::absolute(out_base).string();
        const std::string dat_abs = out_base_abs + ".dat";
        const std::string head_abs = out_base_abs + ".txt";

        std::ostringstream cmd;
        cmd << "\"" << g_path_plot_scattering << "\""
            << " --filename \"" << dat_abs << "\""
            << " --header \"" << head_abs << "\""
            << " --output \"" << out_base_abs << "\""
            << " --theta-i " << theta_i      // required by your build
            << " --phi-i " << phi_i        // required by your build
            << " --format " << 2            // HDR
            << " --image-size " << image_size;

        run_cmd_log(cmd.str(), "vg_tmp/plot.log");
    }

    return out_base;
}

// ---------- public API ----------
// Pure analytic L1 on an (H,W,3) dir grid; NO external tools here.
static pybind11::array_t<float> eval_microfacet_L1_img(
    const std::vector<float>& Dx,
    const std::vector<float>& Dy,
    float cos_ti, float sin_ti,
    float cos_pi, float sin_pi,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> dirs_ow)
{
    namespace py = pybind11;

    // incoming direction (unit)
    const float wi_x = sin_ti * cos_pi;
    const float wi_y = sin_ti * sin_pi;
    const float wi_z = cos_ti;

    // sanity: only upper hemisphere contributes
    if (wi_z <= 0.0f) {
        auto buf = dirs_ow.request();
        if (buf.ndim != 3 || buf.shape[2] != 3) {
            throw std::runtime_error("dirs_ow must be (H,W,3)");
        }
        const int H = static_cast<int>(buf.shape[0]);
        const int W = static_cast<int>(buf.shape[1]);
        py::array_t<float> out({ H, W });
        std::fill((float*)out.request().ptr, (float*)out.request().ptr + H * W, 0.0f);
        return out;
    }

    // map dirs_ow
    auto d_buf = dirs_ow.request();
    if (d_buf.ndim != 3 || d_buf.shape[2] != 3) {
        throw std::runtime_error("dirs_ow must be (H,W,3)");
    }
    const int H = static_cast<int>(d_buf.shape[0]);
    const int W = static_cast<int>(d_buf.shape[1]);
    const float* dptr = static_cast<const float*>(d_buf.ptr);

    // construct NDF
    PiecewiseLinearNDF ndf(Dx, Dy);

    // output array
    py::array_t<float> out({ H, W });
    auto o_buf = out.request();
    float* optr = static_cast<float*>(o_buf.ptr);

    // helper lambdas
    auto norm3 = [](float x, float y, float z) {
        float n = std::sqrt(x * x + y * y + z * z);
        if (n <= 0.0f) return std::array<float, 3>{0.0f, 0.0f, 0.0f};
        return std::array<float, 3>{x / n, y / n, z / n};
        };
    auto dot3 = [](float ax, float ay, float az, float bx, float by, float bz) {
        return ax * bx + ay * by + az * bz;
        };

    // microfacet constants
    const float eps = 1e-6f;
    const float iwz = wi_z;

    // loop pixels
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            const int idx3 = (i * W + j) * 3;
            const float wo_x = dptr[idx3 + 0];
            const float wo_y = dptr[idx3 + 1];
            const float wo_z = dptr[idx3 + 2];

            // upper hemisphere only
            if (wo_z <= 0.0f) {
                optr[i * W + j] = 0.0f;
                continue;
            }

            // half-vector
            const auto hN = norm3(wi_x + wo_x, wi_y + wo_y, wi_z + wo_z);
            const float hx = hN[0], hy = hN[1], hz = hN[2];
            if (hz <= 0.0f) {
                optr[i * W + j] = 0.0f;
                continue;
            }

            // D(h) and Smith masking terms G1(wi,h), G1(wo,h)
            const Vec3f Hv{ hx, hy, hz };
            const Vec3f Wi{ wi_x, wi_y, wi_z };
            const Vec3f Wo{ wo_x, wo_y, wo_z };

            const float D = ndf.D(Hv);
            const float G1i = ndf.G1(Wi, Hv);
            const float G1o = ndf.G1(Wo, Hv);

            // Cook-Torrance single-scatter without Fresnel (F=1).
            // f_r = D(h) * G1(wi,h) * G1(wo,h) / (4 (n.wi) (n.wo))
            const float denom = 4.0f * std::max(eps, iwz * wo_z);
            float fr = (D * G1i * G1o) / denom;
            if (!std::isfinite(fr) || fr < 0.0f) fr = 0.0f;

            // return BRDF value as image (consistent scale for L2+ = Linf - L1 downstream)
            optr[i * W + j] = fr;
        }
    }
    return out;
}


static py::array_t<float> virtual_goniometer_sample(
    const std::vector<float>& Dx,
    const std::vector<float>& Dy,
    float cos_ti, float sin_ti,
    float cos_pi, float sin_pi,
    int H, int W,
    int /*spp*/, int /*max_depth*/, unsigned int /*seed*/)
{
    const std::string out_base = run_pipeline_hdr(Dx, Dy, cos_ti, sin_ti, cos_pi, sin_pi, H, W);

    int h1 = 0, w1 = 0, h2 = 0, w2 = 0;
    std::vector<float> L1s, L2p;
    if (!load_hdr_L(out_base + "_simu_L1.hdr", h1, w1, L1s))
        throw std::runtime_error("Failed to load simulated L1 HDR image.");
    if (!load_hdr_L(out_base + "_simu_L2p.hdr", h2, w2, L2p))
        throw std::runtime_error("Failed to load simulated L2+ HDR image.");
    if (h1 != h2 || w1 != w2) throw std::runtime_error("Simulated images size mismatch.");

    std::vector<float> Linf_img(h1 * w1, 0.0f);
    for (int i = 0; i < h1* w1; ++i) {
        float v = L1s[i] + L2p[i];
        Linf_img[i] = (v > 0.f ? v : 0.f);
    }
    return resample_theta_phi(Linf_img, H, W, h1, w1);
}

// ---------- pybind11 module ----------
PYBIND11_MODULE(ndf_py, m) {
    m.doc() = "Bindings: analytic L1 and virtual goniometer (CLI, HDR).";

    m.def("set_exe_paths", [](const std::string& d2s, const std::string& gonio, const std::string& plot) {
        g_path_distrib_to_surface = d2s;
        g_path_virtual_gonio = gonio;
        g_path_plot_scattering = plot;
        probe_d2s_stats_flag();
        });

    py::class_<PiecewiseLinearNDF>(m, "PiecewiseLinearNDF")
        .def(py::init<const std::vector<float>&, const std::vector<float>&>())
        .def("G1", [](const PiecewiseLinearNDF& self, const Vec3f& w, const Vec3f& h) {
        return self.G1(w, h);
            });

    py::class_<Vec3f>(m, "Vec3f")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vec3f::x)
        .def_readwrite("y", &Vec3f::y)
        .def_readwrite("z", &Vec3f::z);

    m.def("eval_microfacet_L1_img", &eval_microfacet_L1_img,
        py::arg("Dx"), py::arg("Dy"),
        py::arg("cos_ti"), py::arg("sin_ti"),
        py::arg("cos_pi"), py::arg("sin_pi"),
        py::arg("dirs_ow"),
        py::arg("F0") = 1.0f, py::arg("use_smith") = true,
        "Return analytic single-bounce L1 image on (H,W).");

    m.def("virtual_goniometer_sample", &virtual_goniometer_sample,
        py::arg("Dx"), py::arg("Dy"),
        py::arg("cos_ti"), py::arg("sin_ti"),
        py::arg("cos_pi"), py::arg("sin_pi"),
        py::arg("H"), py::arg("W"),
        py::arg("spp") = 128, py::arg("max_depth") = 50, py::arg("seed") = 0,
        "Return total scattering Linf image (Simu L1 + Simu L2+), resampled on (H,W).");
}
