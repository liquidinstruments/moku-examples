<template>
    <ParentLayout>
        <div class="gradient-beam"></div>
        <img
            class="grid-pattern"
            src="../assets/img/lines.svg"
            alt="Grid pattern"
        />

        <section class="hero">
            <div class="hero-intro">
                <h1><span> Moku</span> API Documentation</h1>

                <p class="subtitle">Welcome to the Moku API documentation.</p>

                <p>
                    Here you'll find all you need to get started with the Moku
                    Scripting API, Moku Cloud Compile and the Moku CLI.
                </p>
            </div>
            <div class="hero-cta">
                <div class="hero-cta-text large">
                    <span>
                        <PhHandWaving :size="24" color="#0067FF" />
                        New to Moku?</span
                    >
                    <p>
                        Meet the Moku family of
                        <a
                            href="https://liquidinstruments.com/products/hardware-platforms/"
                            target="_blank"
                            >hardware solutions</a
                        >.
                    </p>
                </div>
                <hr />
                <div class="hero-cta-text">
                    <span>
                        <PhInfo :size="24" color="#0067FF" />
                        Need help?</span
                    >
                    <p>
                        <a
                            href="https://liquidinstruments.com/support/contact/"
                            target="_blank"
                            >Contact support</a
                        >
                        or visit the
                        <a
                            href="https://knowledge.liquidinstruments.com/"
                            target="_blank"
                            >knowledge base</a
                        >.
                    </p>
                </div>
            </div>
        </section>
        <section>
            <h2>Documentation</h2>
            <card-group>
                <card title="Rest API" icon="api" link="/api/">
                    <p>
                        Everything you need to integrate your Moku into your
                        current system with API integration for Python, MATLAB,
                        and LabVIEW
                    </p>
                </card>

                <card title="Moku Cloud Compile" icon="mcc" link="/mcc/">
                    <p>
                        Code, compile, and deploy directly from your browser.
                        Moku Cloud Compile makes FPGA programming simple with
                        the ability to implement custom functionality to
                        Moku:Go, Moku:Lab, and Moku:Pro.Moku:Pro.
                    </p>
                </card>

                <card title="Moku CLI" icon="cli" link="/cli/">
                    <p>
                        Enable command-line access to functions like discovering
                        Moku devices, uploading and downloading files, and
                        converting files.
                    </p>
                </card>

                <card title="Moku Neural Network" icon="mnn" link="/mnn/">
                    <p>
                        The Moku Neural Network instrument facilitates running
                        feedforward, multi-layer neural networks, seamlessly
                        handling both serial and parallel input signals for
                        efficient processing.
                    </p>
                </card>
            </card-group>
        </section>
        <section class="about-wrapper">
            <div class="about">
                <p>
                    At Liquid Instruments we want to help engineers and
                    scientists build technology to solve the world’s greatest
                    challenges.
                </p>
                <action-button
                    text="Learn More"
                    link="https://liquidinstruments.com"
                    target="_blank"
                />
            </div>
        </section>
    </ParentLayout>
</template>

<script>
import ParentLayout from '@parent-theme/layouts/Layout.vue'
import NavLink from '@theme/components/NavLink.vue'
import Navbar from '@theme/components/Navbar.vue'
import SideBar from '@theme/components/Sidebar.vue'
import { PhInfo, PhHandWaving } from 'phosphor-vue'
export default {
    name: 'Landing',
    components: { NavLink, Navbar, SideBar, PhInfo, PhHandWaving },
    computed: {
        data() {
            return this.$page.frontmatter
        },
    },
}
</script>

<style scoped lang="stylus">

#app::has(.landing) {
    background: #FFF;
}

.grid-pattern {
    position: absolute;
    top: -2rem;
    right: 0;
    mix-blend-mode: color-burn;
    transform: skew(-0.02turn, -3deg);
    opacity: 0.2;
}

.gradient::before {
    content: '';
    left: 60%;
    top: -490px;
    position: absolute;
    width: 100px;
    height: 1000px;
    transform: rotate(264deg);
    border-radius: 100px;
    background: linear-gradient(
        -180deg,
        rgba(63, 220, 145, 0.5) 0%,
        #429edb 36.46%,
        rgba(0, 126, 180, 0.7) 76.56%,
        #429edb 100%
    );
    filter: blur(50px);
}

.gradient-beam {
    position: absolute;
    width: 180%;
    height: 95%;
    top: 0;
    left: 0;
    opacity: 0.3;
    transform: translateY(57px);
    mix-blend-mode: hard-light;
    background: linear-gradient(
            279deg,
            rgba(255, 255, 255, 0) 53.86%,
            rgba(3, 82, 115, 0) 58.06%,
            #035273 61.1%,
            rgba(255, 255, 255, 0) 68.34%
        ),
        conic-gradient(
            from 25.82deg at 51.45% 9.73%,
            rgba(255, 255, 255, 0) 0deg,
            rgba(163, 228, 255, 0) 104.4deg,
            #fff0 180deg,
            rgba(255, 255, 255, 0) 360deg
        ),
        #fff0;


    @media screen and (min-width: 768px) {
        left: 25%;
        width: 100%
        background: linear-gradient(
        290deg,
        rgba(255, 255, 255, 0) 53.86%,
        rgba(3, 82, 115, 0) 58.06%,
        #035273 61.1%,
        rgba(255, 255, 255, 0) 68.34%
        ),
        conic-gradient(
            from 25.82deg at 51.45% 9.73%,
            rgba(255, 255, 255, 0) 0deg,
            rgba(163, 228, 255, 0) 104.4deg,
            #fff0 180deg,
            rgba(255, 255, 255, 0) 360deg
        ),
        #fff0;
    }

    @media screen and (min-width: 960px) {
        left: 50%;
    }
}

.gradient-beam::before {
    content: '';
    position: absolute;
    width: 2000px;
    height: 20%;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(180deg, rgba(217, 217, 217, 0) 0%, #fff 100%);
    z-index: 1;
    // @media screen and (min-width: 1024px) {
    //     bottom: -100px;
    // }


    // @media screen and (min-width: 1100px) {
    //     bottom: -200px;
    // }

    @media screen and (min-width: 960px) {
        left: -50%;
    }
}

.theme-container.landing .theme-default-content {
    max-width: $maxWidth;
    margin-top: $headerHeight;
    padding-bottom: 10rem;
}

.theme-default-content:not(.custom) a:hover {
    text-decoration: none;
}

.landing section + section {
    margin-top: 2rem;
    padding-top: 2rem;

    @media screen and (min-width: 768px) {
        margin-top: 4rem;
        padding-top: 4rem;
    }
}

.landing h2 {
    border: none;
    margin-top: 0;
}

.hero {
    margin-top: 4rem;
    display: grid;
    grid-template-columns: repeat(1, minmax(0, 1fr));
    gap: 2rem;
    width: 100%;

    @media screen and (min-width: 1024px) {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 4rem;
    }
}


.hero h1 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    display: inline-block;
    font-weight: 300;
    line-height: 1;
}

.hero h1 span {
    font-weight: 700;
}

.hero p {
    font-size: 1.2rem;
    max-width: 500px;
}

.hero-cta {
    position: relative;
    align-self: start;
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(72, 184, 231, 44%);
    padding: 1rem 1.5rem;
    border-radius: 20px;
    z-index: 5;

    @media screen and (min-width: 1024px) {
        justify-self: center;
    }
}

.hero-cta-text,
.hero-cta span {
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    column-gap: 0.5rem;

    @media screen and (min-width: 768px) {
        flex-direction: row;
        align-items: center;
    }
}


.hero-cta-text,
.hero-cta-text p {
    font-size: 0.9rem;
    margin: .25rem 0;
}

.hero-cta-text span {
    font-weight: 600;
}

.hero-cta-text:not(.large) span {
    gap: 0.5rem;
}

.hero-cta-text.large{
    flex-direction: column;
    align-items: flex-start;
}

.hero-cta-text.large p {
    @media screen and (min-width: 768px) {
        margin-left: calc(24px + 0.5rem);
    }
}

.hero-cta-text.large,
.hero-cta-text.large p {
    font-size: 1rem;
}

.hero-cta p a {
    border-bottom: 1px solid $accentColor;
}

section:has(.about) {
    background: $background;
    border-radius: 1rem;
    border: 1px solid $borderColor;
    padding: 2rem !important;
    margin-top: 4rem !important;

    @media screen and (min-width: 768px) {
        margin-top: 8rem !important
    }
}

.about-wrapper {
    width: fit-content
}

.about {
    background: $textColor;
    color: #FFF;
    padding: 2rem;
    max-width: $maxWidthPara;
    border-radius: .5rem;
    position: relative;
    z-index: 5;
    p {
        margin: 0;
    }

    p:not(.action) {
        font-size: 1.25rem;
        margin-bottom: 1.5rem;
    }

    p,a {
        position: relative;
        z-index: 10;
    }
}
</style>
