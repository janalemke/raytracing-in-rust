use rayon::prelude::*;
use std::io::Write;

#[derive(Default, Copy, Clone, Debug)]
pub struct Vector3(f64, f64, f64);

impl Vector3 {
    pub fn random() -> Self {
        Vector3(rand::random(), rand::random(), rand::random())
    }

    pub fn random_in_range(min: f64, max: f64) -> Self {
        Vector3(
            random_in_range(min, max),
            random_in_range(min, max),
            random_in_range(min, max),
        )
    }

    pub fn random_in_unit_sphere() -> Self {
        loop {
            let p = Vector3::random_in_range(-1.0, 1.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    pub fn random_unit_vector() -> Self {
        Self::random_in_unit_sphere().normalize()
    }

    pub fn random_in_unit_disk() -> Self {
        loop {
            let p = Vector3(random_in_range(-1.0, 1.0), random_in_range(-1.0, 1.0), 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    pub fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x().abs() < s && self.y().abs() < s && self.z().abs() < s
    }

    pub fn reflect(&self, n: &Vector3) -> Vector3 {
        *self - (*n * self.dot(n) * 2.0)
    }

    pub fn refract(&self, n: &Vector3, etai_over_etat: f64) -> Vector3 {
        let cos_theta = (*self * -1.0).dot(n).min(1.0);
        let r_out_perp = (*self + *n * cos_theta) * etai_over_etat;
        let r_out_parallel = *n * (1.0 - r_out_perp.length_squared()).abs().sqrt() * -1.0;
        r_out_perp + r_out_parallel
    }

    pub fn x(&self) -> f64 {
        self.0
    }
    pub fn y(&self) -> f64 {
        self.1
    }
    pub fn z(&self) -> f64 {
        self.2
    }

    pub fn length_squared(&self) -> f64 {
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z()
    }

    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn dot(&self, rhs: &Self) -> f64 {
        self.0 * rhs.0 + self.1 * rhs.1 + self.2 * rhs.2
    }

    pub fn cross(&self, rhs: &Self) -> Self {
        Self(
            self.1 * rhs.2 - self.2 * rhs.1,
            self.2 * rhs.0 - self.0 * rhs.2,
            self.0 * rhs.1 - self.1 * rhs.0,
        )
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        self / len
    }

    fn write_color(&self, mut f: impl Write, samples_per_pixel: usize) {
        let scale = 1.0 / samples_per_pixel as f64;
        let r = (self.0 * scale).sqrt();
        let g = (self.1 * scale).sqrt();
        let b = (self.2 * scale).sqrt();
        writeln!(
            f,
            "{} {} {}",
            (256 as f64 * r.clamp(0.0, 0.999)) as u8,
            (256 as f64 * g.clamp(0.0, 0.999)) as u8,
            (256 as f64 * b.clamp(0.0, 0.999)) as u8
        );
    }
}

impl std::ops::Add for Vector3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl std::ops::AddAssign for Vector3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl std::ops::Sub for Vector3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl std::ops::SubAssign for Vector3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl std::ops::Mul for Vector3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2)
    }
}

impl std::ops::Mul<f64> for Vector3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl std::ops::MulAssign<f64> for Vector3 {
    fn mul_assign(&mut self, rhs: f64) {
        *self = Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl std::ops::Div<f64> for Vector3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs, self.1 / rhs, self.2 / rhs)
    }
}

impl std::ops::DivAssign<f64> for Vector3 {
    fn div_assign(&mut self, rhs: f64) {
        *self = Self(self.0 / rhs, self.1 / rhs, self.2 / rhs)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,
}

impl Ray {
    pub fn at(&self, t: f64) -> Vector3 {
        self.origin + self.direction * t
    }
    pub fn ray_color(&self, world: &impl Hittable, depth: usize) -> Vector3 {
        if depth <= 0 {
            Vector3(0.0, 0.0, 0.0)
        } else {
            if let Some(i) = world.hit(self, 0.001, f64::INFINITY) {
                if let Some((attenuation, scattered)) = i.material.scatter(self, i) {
                    attenuation * scattered.ray_color(world, depth - 1)
                } else {
                    Vector3(0.0, 0.0, 0.0)
                }
            } else {
                let unit_direction = self.direction.normalize();
                let t = (unit_direction.y() + 1.0) * 0.5;
                Vector3(1.0, 1.0, 1.0) * (1.0 - t) + Vector3(0.5, 0.7, 1.0) * t
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Intersection {
    p: Vector3,
    normal: Vector3,
    material: Material,
    t: f64,
    front_facing: bool,
}

impl Intersection {
    fn set_face_normal(&mut self, r: &Ray, outward_normal: &Vector3) {
        self.front_facing = r.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_facing {
            *outward_normal
        } else {
            *outward_normal * -1.0
        }
    }
}

pub trait Hittable: Sync {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Intersection>;
}

#[derive(Debug, Copy, Clone)]
struct Sphere {
    center: Vector3,
    radius: f64,
    material: Material,
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Intersection> {
        let oc = r.origin - self.center;
        let a = r.direction.length_squared();
        let hb = oc.dot(&r.direction);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = hb * hb - a * c;
        if discriminant < 0.0 {
            None
        } else {
            let sd = discriminant.sqrt();
            let mut root = (-hb - sd) / a;
            if root < t_min || t_max < root {
                root = (-hb + sd) / a;
                if root < t_min || t_max < root {
                    return None;
                }
            }
            let mut i = Intersection {
                p: r.at(root),
                normal: (r.at(root) - self.center) / self.radius,
                material: self.material,
                t: root,
                front_facing: false,
            };
            let outward_normal = i.normal;
            i.set_face_normal(r, &outward_normal);
            Some(i)
        }
    }
}

#[derive(Default)]
struct HittableStore {
    objects: Vec<Box<dyn Hittable>>,
}

impl HittableStore {
    fn add(&mut self, object: impl Hittable + 'static) {
        self.objects.push(Box::new(object));
    }

    fn random() -> Self {
        let mut world = Self::default();
        let ground_material = Material::Lambertian {
            albedo: Vector3(0.5, 0.5, 0.5),
        };
        world.add(Sphere {
            center: Vector3(0.0, -1000.0, 0.0),
            radius: 1000.0,
            material: ground_material,
        });

        for a in -11..11 {
            for b in -11..11 {
                let choose_mat = random();
                let center = Vector3(a as f64 + 0.9 * random(), 0.2, b as f64 + 0.9 * random());

                if (center - Vector3(4.0, 0.2, 0.0)).length() > 0.9 {
                    if choose_mat < 0.8 {
                        //diffuse
                        let albedo = Vector3::random() * Vector3::random();
                        let material = Material::Lambertian { albedo };
                        world.add(Sphere {
                            center,
                            radius: 0.2,
                            material,
                        })
                    } else if choose_mat < 0.95 {
                        //metal
                        let albedo = Vector3::random_in_range(0.5, 1.0);
                        let fuzz = random_in_range(0.0, 0.5);
                        let material = Material::Metal { albedo, fuzz };
                        world.add(Sphere {
                            center,
                            radius: 0.2,
                            material,
                        })
                    } else {
                        let material = Material::Dielectric { ir: 1.5 };
                        world.add(Sphere {
                            center,
                            radius: 0.2,
                            material,
                        })
                    }
                }
            }
        }

        let m1 = Material::Dielectric { ir: 1.5 };
        world.add(Sphere {
            center: Vector3(0.0, 1.0, 0.0),
            radius: 1.0,
            material: m1,
        });
        let m2 = Material::Lambertian {
            albedo: Vector3(0.2, 0.2, 0.5),
        };
        world.add(Sphere {
            center: Vector3(-4.0, 1.0, 0.0),
            radius: 1.0,
            material: m2,
        });
        let m3 = Material::Metal {
            albedo: Vector3(0.7, 0.6, 0.5),
            fuzz: 0.2,
        };
        world.add(Sphere {
            center: Vector3(4.0, 1.0, 0.0),
            radius: 1.0,
            material: m3,
        });
        world
    }
}

impl Hittable for HittableStore {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Intersection> {
        self.objects
            .iter()
            .fold((None, t_max), |(i, closest), h| {
                if let Some(ni) = h.hit(r, t_min, closest) {
                    (Some(ni), ni.t)
                } else {
                    (i, closest)
                }
            })
            .0
    }
}

struct Camera {
    origin: Vector3,
    lower_left: Vector3,
    horizontal: Vector3,
    vertical: Vector3,
    u: Vector3,
    v: Vector3,
    w: Vector3,
    lens_radius: f64,
}

impl Camera {
    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = Vector3::random_in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.x() + self.v * rd.y();
        Ray {
            origin: self.origin + offset,
            direction: self.lower_left + self.horizontal * s + self.vertical * t
                - self.origin
                - offset,
        }
    }

    pub fn new(
        lookfrom: Vector3,
        lookat: Vector3,
        vup: Vector3,
        vfov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = vfov / 360.0 * 2.0 * 3.1415926;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).normalize();
        let u = vup.cross(&w).normalize();
        let v = w.cross(&u);

        let origin = lookfrom;
        //let focal_length = 1.0;

        let horizontal = u * viewport_width * focus_dist;
        let vertical = v * viewport_height * focus_dist;
        let lower_left = origin - horizontal / 2.0 - vertical / 2.0 - w * focus_dist;
        let lens_radius = aperture / 2.0;
        Self {
            origin,
            lower_left,
            horizontal,
            vertical,
            u,
            v,
            w,
            lens_radius,
        }
    }
}

// impl Default for Camera {
//     fn default() -> Self {
//         let aspect_ratio = 16.0 / 9.0;
//         let viewport_height = 2.0;
//         let viewport_width = aspect_ratio * viewport_height;
//         let focal_length = 1.0;

//         let horizontal = Vector3(viewport_width, 0.0, 0.0);
//         let vertical = Vector3(0.0, viewport_height, 0.0);
//         let origin = Vector3(0.0, 0.0, 0.0);
//         Self {
//             origin,
//             lower_left: origin
//                 - horizontal / 2.0
//                 - vertical / 2.0
//                 - Vector3(0.0, 0.0, focal_length),
//             horizontal,
//             vertical,
//         }
//     }
// }

pub fn random() -> f64 {
    rand::random()
}

pub fn random_in_range(min: f64, max: f64) -> f64 {
    min + (max - min) * random()
}

#[derive(Copy, Clone, Debug)]
enum Material {
    Lambertian { albedo: Vector3 },
    Metal { albedo: Vector3, fuzz: f64 },
    Dielectric { ir: f64 },
}

impl Material {
    pub fn scatter(&self, r_in: &Ray, intersection: Intersection) -> Option<(Vector3, Ray)> {
        match self {
            Material::Lambertian { albedo } => {
                let mut scatter_direction = intersection.normal + Vector3::random_unit_vector();
                if scatter_direction.near_zero() {
                    scatter_direction = intersection.normal;
                }
                let scattered = Ray {
                    direction: scatter_direction,
                    origin: intersection.p,
                };
                Some((*albedo, scattered))
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = r_in.direction.normalize().reflect(&intersection.normal);
                let scattered = Ray {
                    direction: reflected + Vector3::random_in_unit_sphere() * *fuzz,
                    origin: intersection.p,
                };
                if scattered.direction.dot(&intersection.normal) > 0.0 {
                    Some((*albedo, scattered))
                } else {
                    None
                }
            }
            Material::Dielectric { ir } => {
                let attenuation = Vector3(1.0, 1.0, 1.0);
                let refraction_ratio = if intersection.front_facing {
                    1.0 / ir
                } else {
                    *ir
                };
                let unit_direction = r_in.direction.normalize();

                let cos_theta = (unit_direction * -1.0).dot(&intersection.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                let direction =
                    if cannot_refract || reflectance(cos_theta, refraction_ratio) > random() {
                        unit_direction.reflect(&intersection.normal)
                    } else {
                        unit_direction.refract(&intersection.normal, refraction_ratio)
                    };
                let scattered = Ray {
                    direction: direction,
                    origin: intersection.p,
                };
                Some((attenuation, scattered))
            }
        }
    }
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

fn main() {
    // Image
    let aspect_ratio = 16.0 / 9.0;
    let width = 2560;
    let height = (width as f64 / aspect_ratio) as u32;
    let samples_per_pixel = 100;
    let max_depth = 50;

    // World
    let world = HittableStore::random();
    // let r = (3.1415926/ 4.0 as f64).cos();
    // let mut world = HittableStore::default();
    // let material_ground = Material::Lambertian {
    //     albedo: Vector3(0.8, 0.8, 0.0),
    // };
    // let material_center = Material::Lambertian {
    //     albedo: Vector3(0.1, 0.2, 0.5),
    // };
    // let material_left = Material::Dielectric{
    //     ir: 1.5,
    // };
    // let material_right = Material::Metal {
    //     albedo: Vector3(0.8, 0.6, 0.2),
    //     fuzz: 0.0,
    // };
    // world.add(Sphere {
    //     center: Vector3(0.0, -100.5, -1.0),
    //     radius: 100.0,
    //     material: material_ground,
    // });
    // world.add(Sphere {
    //     center: Vector3(0.0, 0.0, -1.0),
    //     radius: 0.5,
    //     material: material_center,
    // });
    // world.add(Sphere {
    //     center: Vector3(-1.0, 0.0, -1.0),
    //     radius: 0.5,
    //     material: material_left,
    // });
    // world.add(Sphere {
    //     center: Vector3(-1.0, 0.0, -1.0),
    //     radius: -0.4,
    //     material: material_left,
    // });
    // world.add(Sphere {
    //     center: Vector3(1.0, 0.0, -1.0),
    //     radius: 0.5,
    //     material: material_right,
    // });

    //Camera
    let lookfrom = Vector3(13.0, 2.0, 3.0);
    let lookat = Vector3(0.0, 0.0, 0.0);
    let vup = Vector3(0.0, 1.0, 0.0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let cam = Camera::new(
        lookfrom,
        lookat,
        vup,
        20.0,
        aspect_ratio,
        aperture,
        dist_to_focus,
    );

    //Render
    println!(
        "P3
{} {}
255",
        width, height
    );
    let image = (0..height)
        .into_par_iter()
        //.into_iter()
        .rev()
        .flat_map(|j| {
            //eprintln!("Scanlines remaining {}", j);
            (0..width)
                .map(|i| {
                    (0..samples_per_pixel)
                        .map(|_| {
                            let u = (i as f64) / (width - 1) as f64;
                            let v = (j as f64) / (height - 1) as f64;
                            let r = cam.get_ray(u, v);
                            r.ray_color(&world, max_depth)
                        })
                        .fold(Vector3(0.0, 0.0, 0.0), |acc, x| acc + x)
                })
                .collect::<Vec<Vector3>>()
        })
        .collect::<Vec<Vector3>>();
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    for color in image {
        color.write_color(&mut lock, samples_per_pixel)
    }
}
